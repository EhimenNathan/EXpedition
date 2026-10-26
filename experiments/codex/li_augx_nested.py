"""Nested fold selection from fixed LiDAR OOF variant counts; no new fitting."""
import json
import sys
from pathlib import Path

ROOT = Path('/home/nathan/Expedition Competition')
sys.path.insert(0, str(ROOT))
from koz3.score import WEIGHTS, CALC

source = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'Codex Work/li_augx_ensemble_audit.json'
data = json.loads(source.read_text())['folds']
folds = sorted(data)
variants = ['seed_ensemble', 'augx', 'equal_blend', 'two_seed_one_augx']


def metric(vary_by_fold, subset):
    num = den = 0.0
    for cls, weight in WEIGHTS.items():
        counts = [data[f][vary_by_fold[f]]['class_counts'][cls] for f in subset]
        tp = sum(x['tp'] for x in counts)
        fp = sum(x['fp'] for x in counts)
        fn = sum(x['fn'] for x in counts)
        if tp + fn == 0:
            continue
        p = tp / (tp + .5 * fp + CALC.epsilon)
        r = tp / (tp + fn + CALC.epsilon)
        f1 = 2 * p * r / (p + r + CALC.epsilon)
        num += weight * f1
        den += weight
    return num / den


def fixed(name, subset):
    return metric({f: name for f in folds}, subset)


chosen = {}
for hold in folds:
    train = [f for f in folds if f != hold]
    selected = max(variants, key=lambda name: fixed(name, train))
    chosen[hold] = selected
    print(hold, selected, 'test', round(fixed(selected, [hold]), 6),
          'seed', round(fixed('seed_ensemble', [hold]), 6))

result = {'choices': chosen,
          'nested_score': metric(chosen, folds),
          'seed_score': fixed('seed_ensemble', folds),
          'fixed_equal_score': fixed('equal_blend', folds),
          'v10_equal_three_score': fixed('two_seed_one_augx', folds)}
destination = source.with_name(source.stem.replace('ensemble_audit', 'nested')+'.json')
destination.write_text(json.dumps(result, indent=2))
print(result)
