"""Audit split integrity, selected checkpoints, and completed result artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import scipy
import sklearn
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / 'evaluation'))
from nextmove_full_vocab import sha, validate_splits, frequency_baseline


def main():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    data = np.load(RUN / 'dataset.npz', allow_pickle=False)
    assert sha(RUN / 'dataset.npz') == protocol['dataset_sha256']
    assert sha(ROOT / protocol['vocabulary_source']) == protocol['vocabulary_sha256']
    assert sha(ROOT / 'evaluation/nextmove_full_vocab.py') == protocol['execution']['runner_current_sha256']
    assert sha(ROOT / 'evaluation/gpu_multinomial.py') == protocol['execution']['gpu_objective_sha256']
    for info in protocol['checkpoints'].values():
        assert sha(ROOT / info['path']) == info['sha256']
    assert data['target_ply'].min() == 6 and data['target_ply'].max() == 200
    assert len(data['y']) == 183581 and len(np.unique(data['groups'])) == 3000
    gradients, epochs, hashes = [], {}, {}
    for seed in range(3):
        meta = protocol['split_details'][str(seed)]
        splits = {part: data[f'{part}_{seed}'] for part in ['train', 'val', 'test']}
        validate_splits(data['groups'], splits)
        for part, ids in splits.items():
            assert len(ids) == meta['positions'][part]
            assert hashlib.sha256(ids.tobytes()).hexdigest() == meta['split_index_sha256'][part]
        train_y, test_y = (data['y'][splits[part]] for part in ['train', 'test'])
        assert frequency_baseline(train_y, test_y, 1932) == meta['majority']
        assert int(np.sum(~np.isin(test_y, train_y))) == meta['test_targets_absent_from_training']
        for name in ['Scratch', 'PMP-mem', 'PMP-no-mem', 'AE', 'SimCLR', 'Ensemble']:
            path = RUN / 'linear' / f'{name}_{seed}.json'
            row = json.loads(path.read_text())
            assert row['seed'] == seed and row['condition'] == name
            assert not row['warnings'] and row['n_test'] == 10000 and row['n_train'] == 32000
            assert row['fitted_classes'] == meta['train_classes']
            gradient = row['optimization']['gradient_inf']
            assert gradient <= 1e-4 and row['optimization']['dtype'] == 'float64'
            gradients.append(gradient)
            fit = np.load(path.with_suffix('.npz'), allow_pickle=False)
            assert np.array_equal(fit['classes'], np.unique(train_y))
            assert all(np.isfinite(fit[key]).all() for key in fit.files)
            assert 0 <= row['test']['top1'] <= row['test']['top5'] <= 1
            hashes[str(path.relative_to(RUN))] = sha(path)
        for name in ['Scratch', 'PMP-mem', 'PMP-no-mem']:
            path = RUN / 'finetune' / f'{name}_{seed}.json'
            row = json.loads(path.read_text())
            selected = path.with_name(f'{name}_{seed}_selected.pt')
            assert row['selected_checkpoint_sha256'] == sha(selected)
            assert len(row['history']) == 8
            best = max(row['history'], key=lambda ep: ep['val']['top1'])
            assert row['best_epoch'] == best['epoch'] and row['best_val_top1'] == best['val']['top1']
            checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == row['best_epoch'] and checkpoint['condition'] == name and checkpoint['seed'] == seed
            assert checkpoint['state_dict']['head.weight'].shape == (1932, 256)
            assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
            assert 0 <= row['test']['top1'] <= row['test']['top5'] <= 1
            epochs[f'{name}_{seed}'] = row['best_epoch']
            hashes[str(path.relative_to(RUN))] = sha(path)
    reference = json.loads((RUN / 'verification/float64_reference/Scratch_0.json').read_text())
    final = json.loads((RUN / 'linear/Scratch_0.json').read_text())
    report = {
        'linear_runs_verified': 18, 'finetune_runs_verified': 9,
        'disjoint_game_splits_verified': 3, 'maximum_final_float64_gradient': max(gradients),
        'selected_epochs': epochs, 'result_sha256': hashes,
        'software': {'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__, 'numpy': np.__version__},
        'scratch0_pure_float64_reference': {
            'reference_test': reference['test'], 'final_test': final['test'],
            'objective_difference': final['optimization']['objective'] - reference['optimization']['objective'],
            'note': 'Both satisfy the same 1e-4 gradient tolerance; finite-tolerance optimizer trajectories need not give identical predictions.'
        }
    }
    (RUN / 'verification/run_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'result_sha256'}, indent=2))


if __name__ == '__main__':
    main()
