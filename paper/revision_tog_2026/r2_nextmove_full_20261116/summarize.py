"""Summarize only complete, converged runs; all scores refer to held-out test."""
from pathlib import Path
import json
import numpy as np

RUN=Path(__file__).resolve().parent
SEEDS=[0,1,2]
LINEAR=['Scratch','PMP-mem','PMP-no-mem','AE','SimCLR','Ensemble']
FINE=['Scratch','PMP-mem','PMP-no-mem']


def stats(values):
    a=np.array(values,dtype=float)*100
    return {'mean_pct':float(a.mean()),'std_pct':float(a.std(ddof=0)), 'per_seed_pct':a.tolist()}


def main():
    p=json.loads((RUN/'protocol.json').read_text())
    result={'protocol':p,'linear':{},'finetune':{},'paired_lifts':{}}
    for kind,names in [('linear',LINEAR),('finetune',FINE)]:
        for name in names:
            rows=[json.loads((RUN/kind/f'{name}_{seed}.json').read_text()) for seed in SEEDS]
            assert all(row['seed']==seed and row['condition']==name for row,seed in zip(rows,SEEDS))
            if kind=='linear':
                assert all(not row['warnings'] for row in rows), 'Review warnings before reporting'
            else:
                assert all(len(row['history'])==8 for row in rows)
                for row in rows:
                    expected=max(row['history'],key=lambda ep:ep['val']['top1'])
                    assert row['best_epoch']==expected['epoch']
            result[kind][name]={key:stats([row['test'][key] for row in rows]) for key in ['top1','top5']}
        for name in ['PMP-mem','PMP-no-mem']:
            pp=np.array(result[kind][name]['top1']['per_seed_pct'])-np.array(result[kind]['Scratch']['top1']['per_seed_pct'])
            result['paired_lifts'][f'{kind}:{name}-Scratch']={'mean_pp':float(pp.mean()),'std_pp':float(pp.std(ddof=0)),'per_seed_pp':pp.tolist()}
    result['baselines']={
        'Uniform':{'top1':stats([1/p['classes']]*3),'top5':stats([5/p['classes']]*3)},
        'Training frequency':{key:stats([p['split_details'][str(seed)]['majority'][key] for seed in SEEDS]) for key in ['top1','top5']}
    }
    (RUN/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Next-move transfer: complete-vocabulary R2 results','',
           'All scores are held-out test accuracy (%), mean ± population standard deviation over three paired game splits/training seeds. Each test partition contains 10,000 positions. The corpus and pretrained checkpoints are reused across seeds.','']
    for kind,title in [('linear','Frozen representations'),('finetune','End-to-end fine-tuning')]:
        lines += [f'## {title}','','| Condition | Top-1 | Top-5 |','|---|---:|---:|']
        rows={**result['baselines'],**result[kind]}
        for name,row in rows.items():
            cols=[f"{row[key]['mean_pct']:.2f} ± {row[key]['std_pct']:.2f}" for key in ['top1','top5']]
            lines.append(f'| {name} | '+' | '.join(cols)+' |')
        lines.append('')
    lines += ['## Paired top-1 differences from scratch','','| Comparison | Mean difference (pp) | SD (pp) |','|---|---:|---:|']
    for name,row in result['paired_lifts'].items():
        lines.append(f"| {name} | {row['mean_pp']:+.2f} | {row['std_pp']:.2f} |")
    lines += ['','No model was selected using test accuracy. Fine-tuning selected each checkpoint by validation accuracy; all eight epochs and selected epoch numbers are recorded.','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
