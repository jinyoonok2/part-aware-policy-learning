"""Resume the frozen expert subset to thousands without mutating its parent.

Reuses the parent's numerical extractor and immutable raw house directories;
new downloads remain private to this expansion. No opaque pickle is executed.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import Counter, defaultdict
import fcntl
import html
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import h5py
import numpy as np

from data_preparation.data_evaluation_reservations import (check_frozen_reservations, exclude_reserved_order,
    freeze_reservations, load_reservations, require_source_disjoint)


def digest(path):
    import hashlib
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_extractor(run):
    from data_preparation.snapshots import load_source_module
    return load_source_module('frozen_expansion_extractor', run / 'source/collect_interaction_pose_dataset.py')


def check_parent(run, study):
    parent = Path(study['parent_dataset'])
    expected = json.loads((run / 'parent_identity.json').read_text())
    for name, sha in expected.items():
        assert digest(parent / name) == sha, f'Parent modified: {name}'
    assert digest(Path(study['validation_root']) / 'valid_trajectory_index.json') == study['validation_index_sha256']
    for name, sha in study['frozen_source_sha256'].items():
        assert digest(run / 'source' / name) == sha, f'Frozen source changed: {name}'
    for split in ('train', 'val'):
        assert digest(run / f'metadata/{split}_pkgs.parquet') == study['metadata_sha256'][split]
    for name, sha in study.get('selection_file_sha256', {}).items():
        assert digest(run / name) == sha, f'Frozen selection/provenance changed: {name}'
    check_frozen_reservations(run, study, json.loads((run / 'selection_order.json').read_text()))


def parent_identity(parent):
    names = ['study.json', 'house_order.json', 'train_trajectories.jsonl', 'pose_targets.jsonl',
             'val_trajectories.jsonl', 'val_pose_targets.jsonl', 'validation_status.json',
             'status.json', 'source-sha256.json', 'audit/dataset_integrity.json']
    names += [str(p.relative_to(parent)) for folder in ('houses', 'packages') for p in sorted((parent / folder).glob('*.json'))]
    return {name: digest(parent / name) for name in names}


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(args):
    config = json.loads(args.config.read_text())
    parent = Path(config['parent_dataset']).resolve()
    previous = json.loads((parent / 'study.json').read_text())
    initial = json.loads((parent / 'status.json').read_text())
    assert initial['state'] == 'complete' and initial['pose_trajectories'] == 503
    assert previous['revision'] == config['source_revision']
    assert config['target_pose_trajectories'] > initial['pose_trajectories']
    assert json.loads((parent / 'audit/dataset_integrity.json').read_text())['state'] == 'passed'
    order = json.loads((parent / 'house_order.json').read_text())
    validation_records = jsonlines(parent / 'val_trajectories.jsonl')
    reservations = load_reservations(args.project, config.get('reservation_registry'))
    require_source_disjoint(jsonlines(parent / 'train_trajectories.jsonl'), reservations, 'Reused parent Train inventory')
    # Official Val is retained as Test. Its reserved houses are intentionally
    # present here, while the Train inventory/raw-reuse checks remain strict.
    require_source_disjoint([dict(house=int(path.name.removeprefix('house_')))
                             for path in (parent / 'raw').glob('part*/train/house_*') if path.is_dir()],
                            reservations, 'Reused parent raw house directories')
    val_houses = {r['house'] for r in validation_records}
    skipped = [dict(rank=h['rank'], house=h['house'], reason='numeric house ID appears in frozen official Val inventory; conservative exclusion')
               for h in order if config['exclude_validation_numeric_house_ids'] and h['house'] in val_houses]
    skipped_final = [dict(rank=h['rank'], house=h['house'], reason='Reserved final-evaluation house ID; exclude across scene families')
                     for h in order if h['house'] in set(reservations['house_ids'])]
    selected = exclude_reserved_order([h for h in order if h['house'] not in {r['house'] for r in skipped}], reservations)
    assert selected[:initial['houses']] == order[:initial['houses']], 'Expansion would remove an initial house'
    density = initial['pose_trajectories'] / initial['houses']
    projection = []
    for target in (500, 1000, 2000, 5000, 10000):
        n = min(math.ceil(target / density), len(selected))
        projection.append(dict(target_eligible_trajectories=target, estimated_houses=n,
                               compressed_bytes=sum(p['size'] for h in selected[:n] for p in h['packages']),
                               advertised_inflated_bytes=sum(p['inflated_size'] for h in selected[:n] for p in h['packages'])))
    cap = selected[:config['maximum_houses']]
    budget = dict(parent_eligible_trajectories=initial['pose_trajectories'], parent_houses=initial['houses'],
                  trajectories_per_parent_house=density, estimates_not_guarantees=projection,
                  maximum_house_prefix_compressed_bytes=sum(p['size'] for h in cap for p in h['packages']),
                  maximum_house_prefix_inflated_bytes=sum(p['inflated_size'] for h in cap for p in h['packages']),
                  total_compressed_budget=config['download_budget_bytes'],
                  compressed_budget_includes_reused_parent_bytes=initial['download_bytes'],
                  maximum_new_compressed_transfer=config['download_budget_bytes'] - initial['download_bytes'])
    assert budget['maximum_house_prefix_compressed_bytes'] <= config['download_budget_bytes']
    assert budget['maximum_house_prefix_inflated_bytes'] <= config['advertised_inflated_budget_bytes']
    args.run.mkdir(parents=True, exist_ok=False)
    for folder in ('source', 'metadata', 'houses', 'packages', 'downloads', 'logs', 'audit'):
        (args.run / folder).mkdir()
    for name in ('train_pkgs.parquet', 'val_pkgs.parquet'):
        assert digest(parent / 'metadata' / name) == previous['metadata_sha256'][name.split('_')[0]]
        shutil.copy2(parent / 'metadata' / name, args.run / 'metadata' / name)
    parent_source_hashes = json.loads((parent / 'source-sha256.json').read_text())
    extractor = parent / 'source/collect_interaction_pose_dataset.py'
    assert digest(extractor) == parent_source_hashes['collect_interaction_pose_dataset.py']
    shutil.copy2(extractor, args.run / 'source/collect_interaction_pose_dataset.py')
    shutil.copy2(Path(__file__), args.run / 'source/expand_interaction_pose_dataset.py')
    shutil.copy2(Path(__file__).with_name('data_evaluation_reservations.py'), args.run / 'source/data_evaluation_reservations.py')
    shutil.copy2(args.project / 'data_preparation/test_pose_data_expansion.py', args.run / 'source/test_pose_data_expansion.py')
    shutil.copy2(args.config, args.run / 'source/expansion_config.json')
    shutil.copy2(args.project / 'data_preparation/slurm/expand_interaction_pose_dataset.sh', args.run / 'source/expand_interaction_pose_dataset.sh')
    for name in ('train_trajectories.jsonl', 'pose_targets.jsonl', 'val_trajectories.jsonl', 'val_pose_targets.jsonl', 'validation_status.json'):
        shutil.copy2(parent / name, args.run / name)
    for folder in ('houses', 'packages'):
        for path in (parent / folder).glob('*.json'):
            shutil.copy2(path, args.run / folder / path.name)
    reused = []
    for part in sorted((parent / 'raw').iterdir()):
        for house in sorted((part / 'train').iterdir()):
            assert house.is_dir()
            target = args.run / 'raw' / part.name / 'train' / house.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(house.resolve(), target_is_directory=True)
            reused.append(dict(link=str(target.relative_to(args.run)), target=str(house.resolve())))
    (args.run / 'validation_reference').symlink_to(Path(previous['validation_root']), target_is_directory=True)
    (args.run / 'audit/previews').symlink_to(parent / 'audit/previews', target_is_directory=True)
    write(args.run / 'parent_identity.json', parent_identity(parent))
    write(args.run / 'reused_raw_houses.json', reused)
    write(args.run / 'house_order.json', order)
    write(args.run / 'selection_order.json', selected)
    write(args.run / 'excluded_validation_house_ids.json', skipped)
    write(args.run / 'excluded_final_evaluation_house_ids.json', skipped_final)
    write(args.run / 'budget_projection.json', budget)
    frozen_reservations = freeze_reservations(args.run, reservations)
    from data_preparation.snapshots import freeze_packages, source_hashes as snapshot_hashes
    freeze_packages(args.project, args.run / 'source')
    study = dict(previous, name=config['name'], parent_dataset=str(parent),
                 parent_pose_trajectories=initial['pose_trajectories'], parent_download_bytes=initial['download_bytes'],
                 initial_target_pose_trajectories=config['target_pose_trajectories'],
                 download_budget_bytes=config['download_budget_bytes'], advertised_inflated_budget_bytes=config['advertised_inflated_budget_bytes'],
                 maximum_houses=config['maximum_houses'], label_status=config['label_status'], scope=config['scope'],
                 checkpoint_report_interval_houses=config['checkpoint_report_interval_houses'],
                 selection_rule='Original hash order and ranks; full houses; retain all initial references; skip frozen Val and reserved final-evaluation numeric house IDs with explicit ledgers',
                 validation_numeric_house_ids=sorted(val_houses),
                 require_released_valid_mask_for_new_files=True,
                 final_evaluation_reservations=frozen_reservations,
                 selection_file_sha256={name:digest(args.run / name) for name in ('house_order.json','selection_order.json','parent_identity.json','reused_raw_houses.json','excluded_validation_house_ids.json','excluded_final_evaluation_house_ids.json','budget_projection.json')},
                 frozen_source_sha256=snapshot_hashes(args.run / 'source'))
    write(args.run / 'study.json', study)
    write(args.run / 'source-sha256.json', study['frozen_source_sha256'])
    check_parent(args.run, study)
    publish(args.run, study, existing_houses(args.run), 'prepared', consolidate=True)
    print(json.dumps(dict(run=str(args.run), target=config['target_pose_trajectories'], budget=budget), indent=2), flush=True)


def existing_houses(run):
    order = json.loads((run / 'selection_order.json').read_text())
    houses = []
    for item in order:
        path = run / f'houses/house_{item["house"]}.json'
        if path.exists():
            value = json.loads(path.read_text())
            assert value['house'] == item['house'] and value['rank'] == item['rank']
            houses.append(value)
        else:
            break
    return houses


def publish(run, study, houses, state, consolidate=False):
    records = [r for h in houses for r in h['records']]
    targets = [t for h in houses for t in h['targets']]
    if consolidate:
        for name, values in (('train_trajectories.jsonl', records), ('pose_targets.jsonl', targets)):
            temp = run / (name + '.tmp')
            temp.write_text(''.join(json.dumps(v, allow_nan=False) + '\n' for v in values))
            temp.replace(run / name)
    selected_ids = {h['house'] for h in houses}
    assert selected_ids.isdisjoint(study['validation_numeric_house_ids'])
    result = dict(state=state, houses=len(houses), downloaded_packages=sum(h['packages'] for h in houses),
                  total_compressed_bytes=sum(h['download_bytes'] for h in houses),
                  new_compressed_transfer_bytes=sum(h['download_bytes'] for h in houses) - study['parent_download_bytes'],
                  source_trajectories=len(records), successful_valid_trajectories=sum(r.get('valid_trajectory_mask', False) and r.get('recorded_success', False) for r in records),
                  pose_trajectories=len(targets), pose_condition_examples=sum(len(t['condition_frames']) for t in targets),
                  target_pose_trajectories=study['initial_target_pose_trajectories'], parent_references_preserved=study['parent_pose_trajectories'],
                  tasks=dict(Counter(t['task_description'] for t in targets)), active_arms=dict(Counter(t['interaction_event']['arm'] for t in targets)),
                  exclusions=dict(Counter(reason for h in houses for reason in h['exclusions'])),
                  validation_trajectories=197, validation_pose_trajectories=192, validation_unchanged=True,
                  numeric_house_overlap=0, trained_model=False, label_status=study['label_status'],
                  indices_last_consolidated_house_count=len(houses) if consolidate else None)
    # Maintain the consolidated count even during lighter progress updates.
    if not consolidate and (run / 'status.json').exists():
        result['indices_last_consolidated_house_count'] = json.loads((run / 'status.json').read_text()).get('indices_last_consolidated_house_count')
    write(run / 'status.json', result)
    links = '<a href="study.json">Protocol</a> · <a href="budget_projection.json">Budget</a> · <a href="pose_targets.jsonl">Train pose index</a> · <a href="val_pose_targets.jsonl">Separate Val pose index</a> · <a href="audit/expansion_integrity.json">Integrity audit</a>'
    (run / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Expanded interaction-pose Train data</title><style>body{font:16px system-ui;margin:40px;max-width:1200px}pre{white-space:pre-wrap}</style><h1>Expanded official Train expert subset</h1><p>Target: 5,000 eligible offline pose references. Original 503 retained; original validation unchanged. No model training.</p><pre>' + html.escape(json.dumps(result, indent=2)) + '</pre><p>' + links + '</p><p>Metadata pose eligibility is not proof of firm grasp or native replay success. Numeric-house separation does not establish geometry uniqueness across source families.</p>')
    return result


def collect(run):
    study = json.loads((run / 'study.json').read_text())
    with (run / '.collection.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another collection process owns this dataset lock')
        check_parent(run, study)
        extraction = load_extractor(run)
        houses = existing_houses(run)
        order = json.loads((run / 'selection_order.json').read_text())
        done = {h['house'] for h in houses}
        result = publish(run, study, houses, 'collecting', consolidate=True)
        for item in order:
            if item['house'] in done:
                continue
            if result['pose_trajectories'] >= study['initial_target_pose_trajectories']:
                break
            assert len(houses) < study['maximum_houses'], 'House cap reached before target'
            projected = result['total_compressed_bytes'] + sum(p['size'] for p in item['packages'])
            assert projected <= study['download_budget_bytes'], 'Compressed transfer cap reached before target'
            prefix_ids = {h['house'] for h in houses} | {item['house']}
            inflated = sum(p['inflated_size'] for h in order if h['house'] in prefix_ids for p in h['packages'])
            assert inflated <= study['advertised_inflated_budget_bytes'], 'Inflated payload cap reached'
            assert shutil.disk_usage(run).free > max(2_000_000_000, 2 * sum(p['inflated_size'] for p in item['packages'])), 'Insufficient free space for next whole house'
            print(f"Collecting house {item['house']} at original rank {item['rank']}; {result['pose_trajectories']}/{study['initial_target_pose_trajectories']} references", flush=True)
            house = dict(house=item['house'], rank=item['rank'], packages=0, download_bytes=0, records=[], targets=[], exclusions=[])
            for entry in item['packages']:
                # Handle a kill between the legacy helper's final directory
                # rename and receipt write without destroying the orphan data.
                raw_house = run / f'raw/part{entry["part"]}/train/house_{item["house"]}'
                receipt = run / f'packages/part{entry["part"]}_house_{item["house"]}.json'
                if raw_house.exists() and not receipt.exists():
                    assert not raw_house.is_symlink()
                    quarantine = run / f'downloads/orphan_part{entry["part"]}_house_{item["house"]}_{time.time_ns()}'
                    raw_house.rename(quarantine)
                    with (run / 'orphaned_packages.jsonl').open('a') as log:
                        log.write(json.dumps(dict(house=item['house'], part=entry['part'], retained_at=str(quarantine.relative_to(run)), reason='Extracted directory without receipt after interrupted collection')) + '\n')
                package = extraction.fetch_package(run, study, item['house'], entry)
                house['packages'] += 1; house['download_bytes'] += package['download_bytes']
                folder = run / f'raw/part{entry["part"]}/train/house_{item["house"]}'
                assert not folder.is_symlink(), 'Refuse to write to parent raw links'
                for h5 in sorted(folder.glob('*.h5')):
                    sha = digest(h5)
                    with h5py.File(h5, 'r') as data:
                        names = sorted((k for k in data if k.startswith('traj_')), key=lambda n:int(n.split('_')[-1]))
                        mask = data['valid_traj_mask'][:] if 'valid_traj_mask' in data else None
                        for name in names:
                            valid = bool(mask[int(name.split('_')[-1])]) if mask is not None else False
                            try:
                                record, target, reason = extraction.inspect_trajectory(run, study, item['house'], entry['part'], h5, data[name], valid, preview=False)
                                record.update(h5_sha256=sha, pose_eligible=target is not None)
                                if mask is None:
                                    record['source_mask_missing'] = True
                                    reason = 'missing released valid_traj_mask'
                                if reason:
                                    record['pose_exclusion'] = reason; house['exclusions'].append(reason)
                                house['records'].append(record)
                                if target:
                                    target['h5_sha256'] = sha; house['targets'].append(target)
                            except Exception as error:
                                house['records'].append(dict(h5=str(h5.relative_to(run)), trajectory=name, source_split='train', house=item['house'], part=entry['part'], h5_sha256=sha, pose_eligible=False, error=str(error)))
                                house['exclusions'].append(type(error).__name__ + ': ' + str(error))
            write(run / f'houses/house_{item["house"]}.json', house)
            houses.append(house); done.add(item['house'])
            consolidate = len(houses) % study['checkpoint_report_interval_houses'] == 0
            result = publish(run, study, houses, 'collecting', consolidate=consolidate)
            print(json.dumps(result), flush=True)
        check_parent(run, study)
        result = publish(run, study, houses, 'complete', consolidate=True)
        assert result['pose_trajectories'] >= study['initial_target_pose_trajectories'], 'Order exhausted before target'
        audit(run, require_complete=True)
        print('COMPLETE ' + json.dumps(result), flush=True)


def audit(run, require_complete=False):
    study = json.loads((run / 'study.json').read_text())
    check_parent(run, study)
    extraction = load_extractor(run)
    if require_complete:
        assert json.loads((run / 'status.json').read_text())['state'] == 'complete'
    train = jsonlines(run / 'pose_targets.jsonl'); validation = jsonlines(run / 'val_pose_targets.jsonl')
    parent_train = jsonlines(Path(study['parent_dataset']) / 'pose_targets.jsonl')
    assert train[:len(parent_train)] == parent_train, 'Original pose references were changed or reordered'
    train_houses = {r['house'] for r in train}; val_houses = {r['house'] for r in validation}
    assert train_houses.isdisjoint(val_houses)
    reused = json.loads((run / 'reused_raw_houses.json').read_text())
    for item in reused:
        path = run / item['link']
        assert path.is_symlink() and path.resolve() == Path(item['target'])
    for name in ('val_trajectories.jsonl', 'val_pose_targets.jsonl', 'validation_status.json'):
        assert digest(run / name) == digest(Path(study['parent_dataset']) / name)
    file_hashes, transforms = {}, 0
    identities = {}
    for split, rows in (('train', train), ('val', validation)):
        identities[split] = set()
        grouped = defaultdict(list)
        for row in rows:
            assert row['source_split'] == split and row['valid_trajectory_mask'] and row['recorded_success']
            key = row['h5_sha256'], row['trajectory']
            assert key not in identities[split], f'Duplicate {split} trajectory'
            identities[split].add(key)
            path = run / row['h5']
            if split == 'val':
                assert path.resolve().is_relative_to(Path(study['validation_root']).resolve())
            elif path.is_relative_to(run / 'raw'):
                allowed = (run / 'raw').resolve(), (Path(study['parent_dataset']) / 'raw').resolve()
                assert any(path.resolve().is_relative_to(root) for root in allowed)
            else:
                raise ValueError(f'Train HDF5 outside raw namespace: {path}')
            grouped[path].append(row)
        for path, values in grouped.items():
            real = path.resolve()
            if real not in file_hashes:
                file_hashes[real] = digest(real)
            with h5py.File(path, 'r') as data:
                for row in values:
                    assert file_hashes[real] == row['h5_sha256']
                    group = data[row['trajectory']]; event = row['interaction_event']; frame = event['target_frame']
                    actual = extraction.transform(group['obs/extra/robot_base_pose'][frame]) @ extraction.transform(group[f'obs/extra/{event["arm"]}_tcp_pose'][frame])
                    saved = extraction.transform(event['measured_tcp_world_xyz_wxyz'])
                    assert np.allclose(actual, saved, atol=1e-6, rtol=0)
                    assert frame <= row['first_success_frame']
                    assert all(extraction.decode(group['obs/extra/task_info'][j])['handle_grasped'] for j in range(frame-2, frame+1))
                    for condition in row['condition_frames']:
                        current = extraction.transform(group['obs/extra/robot_base_pose'][condition['frame']])
                        assert np.allclose(current @ extraction.transform(condition['target_tcp_in_observation_base_xyz_wxyz']), saved, atol=1e-6, rtol=0)
                        transforms += 1
                    assert all((run / path).is_file() for path in row['videos'].values())
    assert identities['train'].isdisjoint(identities['val'])
    if require_complete:
        assert len(train) >= study['initial_target_pose_trajectories']
    result = dict(state='passed', complete_target_required=require_complete, train_pose_trajectories=len(train), val_pose_trajectories=len(validation),
                  original_references_preserved=len(parent_train), source_hdf5_hashes_checked=len(file_hashes), conditioning_transforms_checked=transforms,
                  duplicate_source_trajectories=0, train_val_source_overlap=0, numeric_house_overlap=0,
                  validation_index_unchanged=True, parent_unchanged=True,
                  source_integrity_passed=True, val_isolation_passed=True,
                  manifest_sha256={name:digest(run / name) for name in ('pose_targets.jsonl','val_pose_targets.jsonl','train_trajectories.jsonl','val_trajectories.jsonl')},
                  geometry_disjointness_not_established_by_numeric_ids=True, physical_replay_validated=False,
                  label_status=study['label_status'], trained_model=False)
    write(run / 'audit/expansion_integrity.json', result)
    write(run / 'audit/dataset_integrity.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','collect','audit','report'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/u/xna8aw/workspace/part-aware-policy-learning'))
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'audit':
        audit(args.run, args.require_complete)
    elif args.command == 'report':
        study = json.loads((args.run / 'study.json').read_text())
        check_parent(args.run, study)
        old = json.loads((args.run / 'status.json').read_text())
        publish(args.run, study, existing_houses(args.run), old['state'], consolidate=True)
    else:
        try:
            collect(args.run)
        except Exception:
            write(args.run / 'collection_error.json', dict(error=traceback.format_exc()))
            study = json.loads((args.run / 'study.json').read_text())
            publish(args.run, study, existing_houses(args.run), 'collection_error', consolidate=True)
            raise


if __name__ == '__main__':
    main()
