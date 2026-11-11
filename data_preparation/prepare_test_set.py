"""Prepare an independent fixed benchmark cohort; never submit execution jobs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.selection import select_bounded_cases
from data_preparation.snapshots import freeze_packages
from grasp.metrics import FINAL_POLICY

PROJECT = Path(__file__).resolve().parents[1]
STORE = Path('/bigtemp/xna8aw/part-aware-policy-learning')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def plan(args):
    """Freeze a new cohort while excluding Train and existing Test houses."""
    if args.run.exists():
        raise FileExistsError(f'Use a new run directory: {args.run}')
    train_index = args.train_dataset / 'train_trajectories.jsonl'
    train_rows = [json.loads(line) for line in train_index.read_text().splitlines() if line.strip()]
    forbidden = {int(row['house']) for row in train_rows}
    forbidden.update(int(path.name.removeprefix('house_'))
                     for path in args.train_dataset.glob('raw/part*/train/house_*'))
    existing_path = args.existing_dataset / 'final_episode_manifest.json'
    existing = json.loads(existing_path.read_text())
    existing_cases = existing.get('cases', existing.get('episodes', []))
    if not isinstance(existing_cases, list) or not existing_cases:
        raise ValueError('Existing Test manifest has no cases')
    forbidden.update(int(row['house']) for row in existing_cases)
    if 2 * args.candidate_houses != args.total:
        raise ValueError('Cohort size must equal two starts per selected house')
    episodes = json.loads(args.benchmark.read_text())
    candidates = [dict(house=episode['house_index'], original_episode_index=index,
        episode_path=f'cases/episode_{index:04d}/episode.json', episode_sha256=canonical(episode),
        case_id=f'episode_{index:04d}') for index, episode in enumerate(episodes)
        if episode['data_split'] == 'val' and episode['house_index'] not in forbidden]
    selected = select_bounded_cases(candidates, episodes, seed=args.seed,
        house_count=args.candidate_houses, batch_size=args.batch_size,
        forbidden_houses=forbidden, smoke_episode_index=-1)
    args.run.mkdir(parents=True)
    cases = []
    for index, row in enumerate(selected['cases']):
        episode_path = args.run / 'cases' / row['case_id'] / 'episode.json'
        write(episode_path, episodes[row['original_episode_index']])
        case = dict(row, episode_path=str(episode_path.relative_to(args.run)),
                    episode_sha256=digest(episode_path), verification_batch=index // args.batch_size + 1,
                    asset_stage_scope=f'batch_{index // args.batch_size + 1:02d}')
        cases.append(case)
        write(episode_path.parent / 'status.json', dict(state='pending',
              case_id=case['case_id'], model_inference_used=False, training_started=False))
    source_hashes = freeze_packages(args.project, args.run / 'source')
    for path in sorted((args.project / 'pose_learning').rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        target = args.run / 'source/pose_learning' / path.relative_to(args.project / 'pose_learning')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(target.relative_to(args.run / 'source'))] = digest(target)
    # Flat entry points support the Slurm worker without importing live code.
    for name in ('verify_benchmark_expert.py', 'report_benchmark_expert_verification.py', 'prepare_smoke.py'):
        target = args.run / 'source' / name
        target.write_bytes((args.project / 'data_preparation' / name).read_bytes())
        source_hashes[name] = digest(target)
    receipt = args.run / 'source/planner_build_receipt.json'
    if args.curobo_receipt.is_file():
        receipt.write_bytes(args.curobo_receipt.read_bytes())
        source_hashes[receipt.name] = digest(receipt)
    study = dict(name='Independent fixed benchmark evaluation cohort',
        project_root=str(args.project.resolve()), source_split='val', project_role='test',
        benchmark=str(args.benchmark.resolve()), benchmark_sha256=digest(args.benchmark),
        train_dataset=str(args.train_dataset.resolve()), train_inventory=str(train_index.resolve()),
        train_inventory_sha256=digest(train_index), train_index=str(train_index.resolve()),
        train_index_sha256=digest(train_index), existing_test_dataset=str(args.existing_dataset.resolve()),
        existing_test_manifest=str(existing_path.resolve()), existing_test_manifest_sha256=digest(existing_path),
        existing_expert_test_trajectories=len(existing_cases), target_qualified_test_episodes=None,
        additional_needed_if_all_existing_qualify=None,
        existing_strict_qualification_verified=False, preserve_all_existing_test_cases=True,
        cases=cases, source_hashes=source_hashes, worker_source_frozen=True, runtime_revision=4,
        smoke_case_index=-1, metric_policy=dict(FINAL_POLICY), metric_version=FINAL_POLICY['metric_version'],
        metric_policy_sha256=canonical(FINAL_POLICY), audit_seed=42, horizon=400,
        maximum_executed_actions=400, continuation_steps=10,
        timing_ms=dict(policy=100., control=20., simulation=4.),
        curobo_revision='417c995647fcb173a2bc094d1284b2a4f4b000ad',
        curobo_build_receipt=str(receipt.resolve()),
        torch_version=importlib.metadata.version('torch'),
        mujoco_version=importlib.metadata.version('mujoco'),
        training_started=False, model_evaluation_started=False,
        automatic_advance_to_later_batches=False,
        cohort_mode='standalone_fixed_starts', target_source_episodes=args.total,
        existing_cases_included=0, witness_format='benchmark_expert_witness_v2',
        primary_population='All fixed starts, including expert failures and unresolved attempts',
        replace_failed_experts=False, source_outcomes_used_for_selection=False,
        excluded_existing_test_house_ids=sorted({int(row['house']) for row in existing_cases}),
        qualification_scope='Separate benchmark cohort. All selected starts remain in the primary '
            'population; expert success and pose-label eligibility are recorded independently. '
            'Original 197-case Test is preserved and is not included in this cohort.')
    write(args.run / 'study.json', study)
    write(args.run / 'selection_plan.json', dict(selected, total_test_target=args.total,
        cohort_mode='standalone_fixed_starts',
        existing_test_trajectories=len(existing_cases), candidate_pool_size=len(cases),
        additional_needed_if_all_existing_qualify=None))
    write(args.run / 'policy_episode_manifest.json', dict(cases=cases,
        cohort_mode='standalone_fixed_starts', requested_cases=len(cases),
        source_case_count=len(cases), pending_cases=0, diagnostic_subset=False,
        partial_export=False, final_model_test_ready=False,
        expert_trajectory_collection_complete=False,
        selection='Fixed benchmark starts; expert outcomes do not change membership'))
    write(args.run / 'status.json', dict(state='planned_not_executed', total_test_target=args.total,
        existing_test_trajectories=len(existing_cases), candidate_pool_size=len(cases),
        qualified_additional_episodes=0, jobs_submitted=False, model_evaluation_started=False))
    print(json.dumps(dict(run=str(args.run), total_test_target=args.total,
        existing_test_trajectories=len(existing_cases), candidate_pool_size=len(cases),
        additional_needed_if_all_existing_qualify=None, standalone=True), indent=2))


def stage(args):
    study = json.loads((args.run / 'study.json').read_text())
    scope = f'batch_{args.batch:02d}'
    rows = [row for row in study['cases'] if row['asset_stage_scope'] == scope]
    if not rows:
        raise ValueError('No candidates in this batch')
    for relative, expected in study['source_hashes'].items():
        if digest(args.run / 'source' / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    output, receipt = args.run / f'assets_{scope}', args.run / f'staged_{scope}.json'
    if receipt.exists():
        if digest(output / 'manifest.json') != json.loads(receipt.read_text())['asset_manifest_sha256']:
            raise ValueError('Staged asset manifest changed')
        return
    indices = [row['original_episode_index'] for row in rows]
    houses = sorted({row['house'] for row in rows})
    subprocess.run([sys.executable, str(args.run / 'source/prepare_smoke.py'), '--output', str(output),
        '--project-root', study['project_root'], '--house-indices', *map(str, houses),
        '--episode-indices', *map(str, indices)], check=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['benchmark_sha256'] != study['benchmark_sha256'] or manifest['original_episode_indices'] != indices:
        raise ValueError('Staged assets belong to different benchmark starts')
    write(receipt, dict(state='passed', scope=scope, candidate_count=len(rows),
        benchmark_sha256=study['benchmark_sha256'], asset_manifest_sha256=digest(output / 'manifest.json'),
        case_episode_sha256={row['case_id']: row['episode_sha256'] for row in rows},
        model_weights_loaded=False, expert_execution_started=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='Metadata and source snapshots only; no GPU jobs')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--train-dataset', type=Path, default=STORE/'datasets/interaction_pose_v2_5000_20261107')
    p.add_argument('--existing-dataset', type=Path, default=STORE/'datasets/interaction_pose_train_test_v1_20261107')
    p.add_argument('--total', type=int, default=400)
    p.add_argument('--candidate-houses', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=6198)
    p.add_argument('--standalone', action='store_true', default=True,
                   help='Independent cohort (the only supported mode; flag is optional)')
    p.add_argument('--project', type=Path, default=PROJECT)
    p.add_argument('--curobo-receipt', type=Path, default=PROJECT/'data_preparation/fixtures/planner_build/curobo_build_receipt.json')
    p = sub.add_parser('stage', help='Stage only one candidate batch; no expert execution')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        if args.total <= 0 or args.candidate_houses <= 0 or args.batch_size <= 0:
            parser.error('Counts must be positive')
        plan(args)
    else:
        stage(args)


if __name__ == '__main__':
    main()
