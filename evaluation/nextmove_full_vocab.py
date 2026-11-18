"""Shared, auditable full-vocabulary transfer protocol for the R2 revision.

Subcommands prepare, embed, linear and finetune write separate resumable files.
No top-K label filtering; held-out games are never used for model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prev_move_models import create_model, fen_to_bitboards
from download_lichess_next import process_game

RUN = ROOT / 'paper/revision_tog_2026/r2_nextmove_full_20261116'
MODELS = {
    'PMP-mem': 'models/prev_move_cnn_v2.pt',
    'PMP-no-mem': 'models/prev_move_cnn_v2_dedup.pt',
    'AE': 'models/ae_cnn.pt',
    'SimCLR': 'models/simclr_cnn_final.pt',
}
SEEDS = [0, 1, 2]
MIN_PLY, MAX_PLY = 6, 200
SIZES = {'train': 32000, 'val': 8000, 'test': 10000}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_vocab():
    rows = [line.split() for line in (ROOT / 'data_next/chess_vocab.txt').read_text().splitlines()]
    vocab = {move: int(idx) for move, idx in rows}
    assert sorted(vocab.values()) == list(range(len(vocab)))
    assert len(vocab) == 1932
    return vocab


def group_splits(groups, seed, sizes=SIZES):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    ordered = rng.permutation(unique)
    # 64/16/20 split by complete game; subsample positions within each partition.
    ntr, nv = int(len(ordered) * .64), int(len(ordered) * .16)
    partitions = {'train': ordered[:ntr], 'val': ordered[ntr:ntr+nv], 'test': ordered[ntr+nv:]}
    result = {}
    for name, game_ids in partitions.items():
        pool = np.flatnonzero(np.isin(groups, game_ids))
        if len(pool) < sizes[name]:
            raise ValueError(f'{name}: only {len(pool)} positions for requested {sizes[name]}')
        result[name] = np.sort(rng.choice(pool, sizes[name], replace=False))
    validate_splits(groups, result)
    return result


def validate_splits(groups, splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            assert not np.intersect1d(splits[left], splits[right]).size
            assert not set(groups[splits[left]]) & set(groups[splits[right]])


def frequency_baseline(train_y, eval_y, n_classes):
    counts = np.bincount(train_y, minlength=n_classes)
    ranking = np.lexsort((np.arange(n_classes), -counts))
    return {'top1': float(np.mean(eval_y == ranking[0])),
            'top5': float(np.mean(np.isin(eval_y, ranking[:5]))),
            'selected_classes': ranking[:5].tolist()}


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'dataset.npz').exists():
        raise FileExistsError('Dataset already prepared; use the recorded dataset.')
    vocab = load_vocab()
    fens, labels, groups, plies, boards = [], [], [], [], []
    game_ids, game_hashes, dates, accepted = [], [], [], 0
    stats = Counter()
    pgn_path = ROOT / 'lichess_data/lichess_2013_01.pgn'
    with pgn_path.open() as stream:
        while accepted < 3000:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            stats['parsed_games'] += 1
            if game.errors:
                stats['parser_error_games'] += 1
                continue
            if game.headers.get('Variant', 'Standard') != 'Standard' or game.headers.get('SetUp', '0') != '0':
                stats['nonstandard_games'] += 1
                continue
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY:
                stats['short_games'] += 1
                continue
            date = game.headers.get('UTCDate', game.headers.get('Date', ''))
            assert date.startswith(('2012.12.31', '2013.01.')), date
            assert game.headers.get('Event', '').startswith('Rated ')
            dates.append(date)
            # Hash full move sequence, so duplicate exports share a partition.
            digest = hashlib.sha256(' '.join(m.uci() for m in moves).encode()).hexdigest()
            gid = len(game_ids)
            game_ids.append(game.headers.get('Site', f'game-{gid}'))
            game_hashes.append(digest)
            pairs = process_game(game, min_ply=MIN_PLY, max_ply=MAX_PLY)
            for ply, (fen, move) in enumerate(pairs, MIN_PLY):
                if move not in vocab:
                    raise ValueError(f'Out-of-vocabulary target {move}: define handling before training')
                board = chess.Board(fen)
                assert chess.Move.from_uci(move) in board.legal_moves
                assert board.ply() == ply - 1
                fens.append(fen); labels.append(vocab[move]); groups.append(digest)
                plies.append(ply); boards.append(board.board_fen())
            accepted += 1
            if accepted % 500 == 0:
                print(f'Prepared {accepted} games, {len(fens)} positions', flush=True)
    y = np.array(labels, dtype=np.int64)
    g = np.array(groups)
    all_fens = np.array(fens)
    payload = {'fens': all_fens, 'y': y, 'groups': g, 'target_ply': np.array(plies, dtype=np.int16)}
    split_meta = {}
    for seed in SEEDS:
        split = group_splits(g, seed)
        for name, idx in split.items():
            payload[f'{name}_{seed}'] = idx
        train_boards = set(np.array(boards)[split['train']])
        test_boards = np.array(boards)[split['test']]
        split_meta[str(seed)] = {
            'positions': {key: len(idx) for key, idx in split.items()},
            'games': {key: len(set(g[idx])) for key, idx in split.items()},
            'train_classes': int(len(np.unique(y[split['train']]))),
            'test_targets_absent_from_training': int(np.sum(~np.isin(y[split['test']], y[split['train']]))),
            'test_piece_placements_also_in_training': int(sum(fen in train_boards for fen in test_boards)),
            'majority': frequency_baseline(y[split['train']], y[split['test']], len(vocab)),
            'split_index_sha256': {key: hashlib.sha256(idx.tobytes()).hexdigest() for key, idx in split.items()},
        }
    np.savez_compressed(RUN / 'dataset.npz', **payload)
    protocol = {
        'source': str(pgn_path.relative_to(ROOT)), 'source_sha256': sha(pgn_path),
        'source_month': '2013-01', 'utc_date_range': [min(dates), max(dates)],
        'games': accepted, 'unique_game_sequences': len(set(game_hashes)),
        'positions': len(fens), 'extraction_counts': dict(stats), 'min_target_ply': MIN_PLY,
        'max_target_ply': MAX_PLY, 'ply_convention': 'target move index, 1-based; board immediately before target',
        'elo_filter': None, 'elo_rationale': 'Retain the original transfer domain (2013 all-rating games); the matching requested ply window is 6-200.',
        'vocabulary_source': 'data_next/chess_vocab.txt', 'vocabulary_sha256': sha(ROOT/'data_next/chess_vocab.txt'),
        'classes': len(vocab), 'observed_classes': int(len(np.unique(y))), 'label_frequency_filter': None,
        'seeds': SEEDS, 'split': '64/16/20 percent of game-sequence groups; sample 32000/8000/10000 positions',
        'split_details': split_meta, 'dataset_sha256': sha(RUN/'dataset.npz'),
        'fine_tune': {'epochs': 8, 'batch_size': 256, 'optimizer': 'Adam', 'lr': 1e-4,
                      'selection': 'highest validation top-1; first epoch on ties; test only after selection',
                      'precision': 'float32, TF32 enabled', 'initialization': 'PyTorch default seeded trunk; identical fresh head for all conditions per seed'},
        'linear': {'estimator': 'LogisticRegression', 'solver': 'lbfgs', 'C': 1., 'max_iter': 2000,
                   'tol': 1e-4, 'preprocessing': 'StandardScaler fitted on training partition only',
                   'absent_training_classes': 'No test rows dropped; test labels absent from fitted classes count as errors'},
        'checkpoints': {name: {'path': path, 'sha256': sha(ROOT/path)} for name,path in MODELS.items()},
        'script_sha256': sha(__file__), 'torch': torch.__version__, 'numpy': np.__version__,
    }
    dump(RUN/'protocol.json', protocol)
    print(json.dumps({'positions': len(fens), 'classes': len(vocab), 'splits': split_meta}, indent=2), flush=True)


def data():
    return np.load(RUN/'dataset.npz', allow_pickle=False)


def encoder(name, seed):
    torch.manual_seed(seed + 100)
    checkpoint = torch.load(ROOT / MODELS.get(name, MODELS['PMP-mem']), map_location='cpu', weights_only=False)
    model = create_model(checkpoint['arch'], checkpoint['n_moves'], embed_dim=checkpoint['embed_dim'])
    if name != 'Scratch':
        model.load_state_dict(checkpoint['model'])
    # Remove unused previous-move head; get_embedding uses only the trunk.
    if hasattr(model, 'head'):
        model.head = nn.Identity()
    return model


def configure_torch():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this scheduled run; no silent CPU fallback')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device('cuda')


def embed():
    device = configure_torch()
    d = data()
    union = np.unique(np.concatenate([d[f'{part}_{seed}'] for seed in SEEDS for part in SIZES]))
    np.save(RUN/'embedding_indices.npy', union)
    features = torch.from_numpy(np.stack([fen_to_bitboards(f) for f in d['fens'][union]])).to(device)
    jobs = [(name, 0) for name in MODELS] + [('Scratch', seed) for seed in SEEDS]
    for name, seed in jobs:
        out = RUN / 'embeddings' / f'{name}_{seed}.npy'
        if out.exists():
            continue
        model = encoder(name, seed).to(device).eval()
        vectors = []
        start = time.time()
        with torch.inference_mode():
            for offset in range(0, len(features), 512):
                vectors.append(model.get_embedding(features[offset:offset+512]).cpu().numpy())
        out.parent.mkdir(exist_ok=True)
        array = np.concatenate(vectors)
        assert array.shape == (len(union), 256) and np.isfinite(array).all()
        np.save(out, array)
        print(f'Embeddings {name} seed={seed}: {array.shape}, {time.time()-start:.1f}s', flush=True)
        del model


def classifier_metrics(clf, X, y):
    p = clf.predict_proba(X)
    order = np.argsort(-p, axis=1, kind='stable')[:, :5]
    labels = clf.classes_[order]
    return {'top1': float(np.mean(labels[:, 0] == y)), 'top5': float(np.mean(np.any(labels == y[:, None], axis=1)))}


def linear(gpu=False):
    if gpu:
        configure_torch()
        from gpu_multinomial import GPUMultinomial
    d = data()
    union = np.load(RUN/'embedding_indices.npy')
    lookup = np.full(len(d['y']), -1, dtype=np.int64)
    lookup[union] = np.arange(len(union))
    names = ['Scratch', 'PMP-mem', 'PMP-no-mem', 'AE', 'SimCLR', 'Ensemble']
    for seed in SEEDS:
        for name in names:
            out = RUN / 'linear' / f'{name}_{seed}.json'
            if out.exists():
                continue
            parts = ['AE', 'SimCLR', 'PMP-mem'] if name == 'Ensemble' else [name]
            arrays = [np.load(RUN/'embeddings'/f'{part}_{seed if part == "Scratch" else 0}.npy', mmap_mode='r') for part in parts]
            train_ids = lookup[d[f'train_{seed}']]
            test_ids = lookup[d[f'test_{seed}']]
            assert np.all(train_ids >= 0) and np.all(test_ids >= 0)
            Xtr = np.concatenate([a[train_ids] for a in arrays], axis=1).astype(np.float64)
            Xtest = np.concatenate([a[test_ids] for a in arrays], axis=1).astype(np.float64)
            ytr, ytest = d['y'][d[f'train_{seed}']], d['y'][d[f'test_{seed}']]
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr); Xtest = scaler.transform(Xtest)
            clf = (GPUMultinomial(C=1., max_iter=2000, tol=1e-4) if gpu else
                   LogisticRegression(C=1., max_iter=2000, tol=1e-4, solver='lbfgs'))
            start = time.time()
            print(f'Fit linear {name} seed={seed}', flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                clf.fit(Xtr, ytr)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f'Nonconverged linear head: {name}, {seed}; do not report incomplete fit')
            metrics = classifier_metrics(clf, Xtest, ytest)
            result = {'condition': name, 'seed': seed, 'test': metrics, 'n_train': len(ytr), 'n_test': len(ytest),
                      'fitted_classes': len(clf.classes_), 'iterations': clf.n_iter_.tolist(),
                      'warnings': [str(w.message) for w in caught], 'seconds': time.time()-start,
                      'optimization': getattr(clf, 'optimization_', {'backend': 'sklearn CPU'})}
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'), coefficients=clf.coef_, intercept=clf.intercept_,
                                classes=clf.classes_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
            dump(out, result)
            print(json.dumps(result), flush=True)


class TransferModel(nn.Module):
    def __init__(self, trunk, n_classes, seed):
        super().__init__()
        self.trunk = trunk
        torch.manual_seed(seed + 1000)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        return self.head(self.trunk.get_embedding(x))


def evaluate(model, X, y, batch=512):
    model.eval()
    hits1 = hits5 = 0
    loss = 0.
    with torch.inference_mode():
        for offset in range(0, len(y), batch):
            yy = y[offset:offset+batch]
            logits = model(X[offset:offset+batch])
            top = logits.topk(5, dim=1).indices
            hits1 += (top[:, 0] == yy).sum().item()
            hits5 += (top == yy[:, None]).any(dim=1).sum().item()
            loss += nn.functional.cross_entropy(logits, yy, reduction='sum').item()
    return {'top1': hits1/len(y), 'top5': hits5/len(y), 'cross_entropy': loss/len(y)}


def finetune():
    device = configure_torch()
    d = data()
    n_classes = len(load_vocab())
    for seed in SEEDS:
        tensors = {}
        for part in SIZES:
            idx = d[f'{part}_{seed}']
            X = torch.from_numpy(np.stack([fen_to_bitboards(fen) for fen in d['fens'][idx]])).to(device)
            y = torch.from_numpy(d['y'][idx]).to(device)
            tensors[part] = (X, y)
        for name in ['Scratch', 'PMP-mem', 'PMP-no-mem']:
            out = RUN/'finetune'/f'{name}_{seed}.json'
            if out.exists():
                continue
            torch.manual_seed(seed + 2000)
            model = TransferModel(encoder(name, seed), n_classes, seed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            generator = torch.Generator().manual_seed(seed + 3000)
            Xtr, ytr = tensors['train']
            history, best_acc, best_epoch, best_state = [], -1., None, None
            started = time.time()
            for epoch in range(1, 9):
                epstart = time.time()
                model.train()
                order = torch.randperm(len(ytr), generator=generator).to(device)
                total_loss = 0.
                for offset in range(0, len(ytr), 256):
                    ids = order[offset:offset+256]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(Xtr[ids]), ytr[ids])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward(); optimizer.step()
                    total_loss += loss.item()*len(ids)
                validation = evaluate(model, *tensors['val'])
                history.append({'epoch': epoch, 'train_loss': total_loss/len(ytr), 'val': validation, 'seconds': time.time()-epstart})
                if validation['top1'] > best_acc:
                    best_acc, best_epoch = validation['top1'], epoch
                    best_state = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
                dump(RUN/'finetune'/f'{name}_{seed}_progress.json', {'history': history, 'best_epoch': best_epoch})
                print(f'Fine-tune {name} seed={seed} epoch={epoch}/8 val={validation["top1"]:.4%}, {time.time()-epstart:.1f}s', flush=True)
            model.load_state_dict(best_state)
            test = evaluate(model, *tensors['test'])
            model_path = RUN/'finetune'/f'{name}_{seed}_selected.pt'
            torch.save({'state_dict':best_state,'seed':seed,'condition':name,'epoch':best_epoch}, model_path)
            result = {'condition':name,'seed':seed,'best_epoch':best_epoch,'best_val_top1':best_acc,
                      'test':test,'history':history,'seconds':time.time()-started,
                      'selected_checkpoint_sha256':sha(model_path)}
            dump(out,result)
            print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
            del model, optimizer, best_state


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','embed','linear','linear-gpu','finetune'])
    args=parser.parse_args()
    if args.command == 'linear-gpu':
        linear(gpu=True)
    else:
        globals()[args.command]()
