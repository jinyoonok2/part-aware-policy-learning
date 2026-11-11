"""Recount the retained 7/10 and 10/10 contact evidence from original traces."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(run):
    policy = json.loads((run / 'audit_policy.json').read_text())
    expected = policy['input_hashes']
    aggregate = {}
    for method in ('reference', 'close20mm_and_base_recovery'):
        final = sustained = 0
        for index in range(10):
            relative = f'cases/{index:02d}/{method}_trace.json'
            blob = (run / relative).read_bytes()
            if hashlib.sha256(blob).hexdigest() != expected[relative]:
                raise ValueError('Original trace identity changed: ' + relative)
            trace = json.loads(blob)
            if len(trace) != 120 or [row['step'] for row in trace] != list(range(1, 121)):
                raise ValueError('The complete 120-step trace is required: ' + relative)
            final += bool(trace[-1]['active_arm_exact_both_fingers'])
            sustained += all(row['active_arm_exact_both_fingers'] for row in trace[-10:])
        aggregate[method] = dict(cases=10, final_bilateral_contact=final,
                                 uninterrupted_final_ten_samples=sustained)
    return dict(scope='Local contact after expert approach; no full-opening success claim',
                source='Saved historical traces; no new rollout', aggregate=aggregate,
                original_worker_errors_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/bigtemp/xna8aw/part-aware-policy-learning/runs/grasp_7to10'))
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))


if __name__ == '__main__':
    main()
