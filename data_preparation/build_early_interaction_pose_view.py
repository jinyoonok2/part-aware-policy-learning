"""Create separate early-grasp pose indices from all source eligible events.

Preserves original raw files/indices. Confidence is metadata-based, not proof of
firm physical grasp. HDF5 files are opened read-only and no pickle is executed.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import html
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    import hashlib
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_lines(path, rows):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temp.replace(path)


def pose_matrix(pose):
    pose = np.asarray(pose)
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    value[:3, 3] = pose[:3]
    return value


def pose7(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)].tolist()


def contiguous(values):
    return [x for x in np.split(values, np.flatnonzero(np.diff(values) != 1) + 1) if len(x)]


def select_event(events, fractions, maximum=.1, require_no_prior=True):
    fractions = np.asarray(fractions, dtype=float)
    assert np.isfinite(fractions).all()
    ordered = sorted(events, key=lambda x: (x['target_frame'], x['start_frame'], x['arm']))
    inspected = []
    choice = None
    for event in ordered:
        frame = event['target_frame']
        opening = float(fractions[frame])
        prior = float(np.max(fractions[:frame + 1]))
        target_early = opening <= maximum + 1e-9
        prefix_early = prior <= maximum + 1e-9
        inspected.append(dict(start_frame=event['start_frame'], target_frame=frame, arm=event['arm'],
                              source_opening_fraction=opening, prior_max_opening_fraction=prior,
                              target_frame_early=bool(target_early), no_prior_major_opening=bool(prefix_early)))
        if choice is None and target_early and (prefix_early or not require_no_prior):
            choice = event
    return choice, inspected


def conditioning_frames(phase, pregrasp_id, start, count=4):
    candidates = np.flatnonzero((np.asarray(phase) == pregrasp_id) & (np.arange(len(phase)) < start))
    approach_start = int(contiguous(candidates)[-1][0]) if len(candidates) else max(0, start - 20)
    assert start > approach_start
    return sorted(set(np.linspace(approach_start, start - 1, count, dtype=int).tolist())), approach_start


def conditioning_records(base_poses, target_world, frames, fractions):
    return [dict(frame=int(i), target_tcp_in_observation_base_xyz_wxyz=pose7(np.linalg.inv(pose_matrix(base_poses[i])) @ target_world),
                 source_door_opening_fraction=float(fractions[i])) for i in frames]


def balance_counts(ledger, key):
    groups = {}
    for row in ledger:
        value = str(row.get(key) if row.get(key) is not None else 'unknown')
        item = groups.setdefault(value,dict(inventory=0,original_pose_eligible=0,early_retained=0,excluded=0,exclusion_reasons={}))
        item['inventory'] += 1
        item['original_pose_eligible'] += int(row['source_pose_eligible'])
        item['early_retained'] += int(row['retained'])
        item['excluded'] += int(not row['retained'])
        if not row['retained']:
            reason=row['reason'];item['exclusion_reasons'][reason]=item['exclusion_reasons'].get(reason,0)+1
    return groups


def prepare(args):
    dataset = args.dataset.resolve()
    source_study = json.loads((dataset / 'study.json').read_text())
    config = json.loads(args.config.read_text())
    args.run.mkdir(parents=True, exist_ok=False)
    for folder in ('source', 'logs'):
        (args.run / folder).mkdir()
    source_hashes = json.loads((dataset / 'source-sha256.json').read_text())
    extractor = dataset / 'source/collect_interaction_pose_dataset.py'
    assert digest(extractor) == source_hashes['collect_interaction_pose_dataset.py']
    shutil.copy2(extractor, args.run / 'source/frozen_source_extractor.py')
    for name in ('data_preparation/build_early_interaction_pose_view.py', 'data_preparation/test_early_pose_quality.py',
                 'data_preparation/slurm/build_early_interaction_pose_view.sh'):
        shutil.copy2(args.project / name, args.run / 'source' / Path(name).name)
    shutil.copy2(args.config, args.run / 'source/quality_config.json')
    from data_preparation.snapshots import freeze_packages, source_hashes as snapshot_hashes
    freeze_packages(args.project, args.run / 'source')
    study = dict(config, dataset_root=str(dataset), dataset_study_sha256=digest(dataset / 'study.json'),
                 minimum_source_pose_trajectories=args.minimum_source_targets,
                 source_split='train and val kept separate',
                 source_revision=source_study['revision'], validation_root=source_study['validation_root'],
                 validation_index_sha256=source_study['validation_index_sha256'],
                 paths_in_quality_indices='h5/videos remain relative to dataset_root, not this quality-view directory',
                 frozen_source_sha256=snapshot_hashes(args.run / 'source'))
    write(args.run / 'study.json', study)
    write(args.run / 'status.json', dict(state='prepared', dataset_root=str(dataset), no_training=True))
    report(args.run)


def source_event(group, scene, event, extraction, source_study, minimum_samples):
    start, end, side = event['start_frame'], event['target_frame'], event['arm']
    assert start > 0 and end - start + 1 >= minimum_samples
    assert np.all(group['obs/extra/policy_phase'][start:end + 1] == int(scene['policy_phases']['GRASP_HANDLE']))
    assert all(extraction.decode(group['obs/extra/task_info'][i])['handle_grasped'] for i in range(end - 2, end + 1))
    before, after = [extraction.decode(group['obs/agent/qpos'][i]) for i in (start - 1, end)]
    closure = float(np.abs(before[f'{side}_gripper']).sum() - np.abs(after[f'{side}_gripper']).sum())
    assert closure >= source_study['minimum_gripper_closure_proxy_m']
    base = extraction.transform(group['obs/extra/robot_base_pose'][end])
    tcp = extraction.transform(group[f'obs/extra/{side}_tcp_pose'][end])
    world = base @ tcp
    assert np.allclose(world, extraction.transform(event['measured_tcp_world_xyz_wxyz']), atol=1e-6, rtol=0)
    return world


def build(run):
    with (run / '.build.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another quality builder owns this view')
        study = json.loads((run / 'study.json').read_text())
        dataset = Path(study['dataset_root'])
        assert digest(dataset / 'study.json') == study['dataset_study_sha256']
        for name, sha in study['frozen_source_sha256'].items():
            assert digest(run / 'source' / name) == sha
        source_status = json.loads((dataset / 'status.json').read_text())
        assert source_status['state'] == 'complete', 'Wait for collection and audit to complete'
        source_audit = json.loads((dataset / 'audit/dataset_integrity.json').read_text())
        assert source_audit['state'] == 'passed'
        assert source_status['pose_trajectories'] >= study['minimum_source_pose_trajectories']
        assert digest(Path(study['validation_root']) / 'valid_trajectory_index.json') == study['validation_index_sha256']
        from data_preparation.snapshots import load_source_module
        extraction = load_source_module('early_view_frozen_extractor', run / 'source/frozen_source_extractor.py')
        source_study = json.loads((dataset / 'study.json').read_text())
        input_names = ('pose_targets.jsonl','val_pose_targets.jsonl','train_trajectories.jsonl','val_trajectories.jsonl')
        input_hashes = {name:digest(dataset / name) for name in input_names}
        if source_audit.get('manifest_sha256'):
            for name, sha in input_hashes.items():
                assert source_audit['manifest_sha256'][name] == sha
        output_by_split, ledgers, statistics = {}, {}, {}
        source_hashes, transform_count = {}, 0
        for split, target_name, inventory_name in (('train','pose_targets.jsonl','train_trajectories.jsonl'),
                                                    ('val','val_pose_targets.jsonl','val_trajectories.jsonl')):
            originals = lines(dataset / target_name)
            inventory = lines(dataset / inventory_name)
            assert all(r['source_split']==split for r in originals+inventory)
            original_lookup = {(r['h5'],r['trajectory']):r for r in originals}
            assert len(original_lookup) == len(originals)
            grouped = defaultdict(list)
            ledger = []
            for row in inventory:
                key = row['h5'], row['trajectory']
                if key in original_lookup:
                    grouped[dataset / row['h5']].append(original_lookup[key])
                else:
                    ledger.append(dict(source_split=split, house=row.get('house'), h5=row['h5'], trajectory=row['trajectory'],
                                       task_description=row.get('task_description'), original_arm=None,
                                       retained=False, source_pose_eligible=False, reason=row.get('pose_exclusion', row.get('error','no original eligible pose event'))))
            outputs = []
            for path, rows in grouped.items():
                actual_sha = digest(path)
                source_hashes[str(path.resolve())] = actual_sha
                with h5py.File(path, 'r') as data:
                    for row in rows:
                        assert row['source_split'] == split and actual_sha == row['h5_sha256']
                        group = data[row['trajectory']]
                        scene = extraction.decode(group['obs_scene'][()])
                        raw_phase = group['obs/extra/policy_phase'][:]
                        fractions = [float(np.asarray(extraction.decode(x)['opening_percentage']).reshape(-1)[0]) for x in group['obs/extra/door_state']]
                        events = sorted(row.get('other_eligible_events', []) + [row['interaction_event']], key=lambda e:e['target_frame'])
                        assert len(events) == row['grasp_event_count']
                        assert len({(e['target_frame'],e['arm']) for e in events}) == len(events)
                        for event in events:
                            source_event(group, scene, event, extraction, source_study, study['minimum_grasp_phase_samples'])
                        selected, inspected = select_event(events, fractions, study['maximum_source_opening_fraction'], study['require_no_prior_major_opening'])
                        original_opening = fractions[row['interaction_event']['target_frame']]
                        record = dict(source_split=split, house=row['house'], part=row['part'], h5=row['h5'], h5_sha256=actual_sha,
                                      task_description=row['task_description'], original_arm=row['interaction_event']['arm'],
                                      trajectory=row['trajectory'], source_pose_eligible=True, retained=selected is not None,
                                      original_target_frame=row['interaction_event']['target_frame'], original_target_opening_fraction=original_opening,
                                      original_target_late=original_opening > study['maximum_source_opening_fraction'],
                                      had_target_early_event=any(e['target_frame_early'] for e in inspected),
                                      had_initial_early_event=any(e['target_frame_early'] and e['no_prior_major_opening'] for e in inspected),
                                      candidate_events=inspected, label_confidence=study['label_confidence'])
                        if selected is None:
                            record['reason'] = 'early target exists only after prior major opening' if record['had_target_early_event'] else 'no eligible grasp event before opening threshold'
                        else:
                            world = source_event(group, scene, selected, extraction, source_study, study['minimum_grasp_phase_samples'])
                            frames, approach_start = conditioning_frames(raw_phase, int(scene['policy_phases']['REACH_PRE_GRASP']), selected['start_frame'], study['conditioning_samples'])
                            conditions = conditioning_records(group['obs/extra/robot_base_pose'][:], world, frames, fractions)
                            for condition in conditions:
                                actual = extraction.transform(group['obs/extra/robot_base_pose'][condition['frame']]) @ extraction.transform(condition['target_tcp_in_observation_base_xyz_wxyz'])
                                assert np.allclose(actual, world, atol=1e-6, rtol=0)
                                transform_count += 1
                            others = [e for e in events if e is not selected]
                            rebuilt = dict(row, interaction_event=selected, other_eligible_events=others,
                                           event_definition=study['label_definition'], condition_frames=conditions,
                                           original_target_frame=row['interaction_event']['target_frame'],
                                           original_event_definition=row['event_definition'],
                                           selected_source_opening_fraction=fractions[selected['target_frame']],
                                           selected_source_prior_max_opening_fraction=max(fractions[:selected['target_frame']+1]),
                                           quality_label_confidence=study['label_confidence'], dataset_root=str(dataset),
                                           quality_view='initial_grasp_before_major_opening_v1')
                            # Do not retain a contact sheet that depicts the old later target.
                            rebuilt.pop('audit_preview',None)
                            outputs.append(rebuilt)
                            record.update(selected_target_frame=selected['target_frame'], selected_arm=selected['arm'],
                                          selected_source_opening_fraction=fractions[selected['target_frame']],
                                          corresponding_approach_start=approach_start, conditioning_frames=frames,
                                          selected_earlier_than_original=selected['target_frame'] < row['interaction_event']['target_frame'],
                                          reason='retained earliest metadata-eligible initial grasp')
                        ledger.append(record)
                assert digest(path) == actual_sha, 'Read-only source file changed during quality audit'
                write(run / 'status.json', dict(state='building', split=split, audited_trajectories=len(ledger), retained_labels=len(outputs), no_training=True))
            outputs.sort(key=lambda r:(r['house'],r['part'],r['h5'],r['trajectory']))
            ledger.sort(key=lambda r:(r.get('house') or -1,r['h5'],r['trajectory']))
            stats = dict(source_inventory_trajectories=len(inventory), original_pose_labels=len(originals), early_pose_labels=len(outputs),
                         conditioning_examples=sum(len(r['condition_frames']) for r in outputs), houses=len({r['house'] for r in outputs}),
                         original_targets_late=sum(r.get('original_target_late',False) for r in ledger),
                         labels_with_target_early_event=sum(r.get('had_target_early_event',False) for r in ledger),
                         labels_with_initial_early_event=sum(r.get('had_initial_early_event',False) for r in ledger),
                         labels_switched_to_earlier_event=sum(r.get('selected_earlier_than_original',False) for r in ledger),
                         late_originals_recovered=sum(r.get('original_target_late',False) and r['retained'] for r in ledger),
                         exclusions=dict(Counter(r['reason'] for r in ledger if not r['retained'])),
                         tasks=dict(Counter(r['task_description'] for r in outputs)),
                         active_arms=dict(Counter(r['interaction_event']['arm'] for r in outputs)),
                         by_task=balance_counts(ledger,'task_description'),
                         by_source_arm=balance_counts(ledger,'original_arm'),
                         by_house=balance_counts(ledger,'house'))
            output_by_split[split], ledgers[split], statistics[split] = outputs, ledger, stats
            write_lines(run / f'{split}_early_pose_targets.jsonl', outputs)
            write_lines(run / f'{split}_quality_ledger.jsonl', ledger)
        train_ids = {(r['h5_sha256'],r['trajectory']) for r in output_by_split['train']}
        val_ids = {(r['h5_sha256'],r['trajectory']) for r in output_by_split['val']}
        assert train_ids.isdisjoint(val_ids)
        assert {r['house'] for r in output_by_split['train']}.isdisjoint({r['house'] for r in output_by_split['val']})
        for name, sha in input_hashes.items():
            assert digest(dataset / name) == sha, f'Original index changed during quality build: {name}'
        assert digest(Path(study['validation_root']) / 'valid_trajectory_index.json') == study['validation_index_sha256']
        result = dict(state='complete', statistics=statistics, conditioning_transforms_checked=transform_count,
                      source_hdf5_hashes_checked=len(source_hashes), source_input_manifest_sha256=input_hashes,
                      output_manifest_sha256={name:digest(run / name) for name in ('train_early_pose_targets.jsonl','val_early_pose_targets.jsonl','train_quality_ledger.jsonl','val_quality_ledger.jsonl')},
                      original_indices_unchanged=True, validation_index_unchanged=True, train_val_source_overlap=0,
                      numeric_house_overlap=0, physical_force_validation=False, label_confidence=study['label_confidence'], no_training=True)
        write(run / 'source_hdf5_sha256.json', source_hashes)
        with (run / 'house_balance.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['split','house','inventory','original_pose_eligible','early_retained','excluded'])
            for split,stats in statistics.items():
                for house,values in stats['by_house'].items():
                    writer.writerow([split,house,*[values[k] for k in ('inventory','original_pose_eligible','early_retained','excluded')]])
        write(run / 'status.json', result)
        report(run)
        print(json.dumps(dict(state='complete', statistics={split:{k:v for k,v in stats.items() if k not in ('by_house','by_task','by_source_arm')} for split,stats in statistics.items()}, conditioning_transforms_checked=transform_count),indent=2),flush=True)


def report(run):
    study = json.loads((run / 'study.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    description = ['# Initial-grasp pose quality view', '',
                   'Additional indices only; original raw data and Train/Val indices are preserved. All HDF5 inputs are read-only. The primary view selects the earliest source-eligible grasp before 10% opening and before any prior >10% opening. Target-frame-only counts are also reported.', '',
                   'Labels are based on recorded phases, closure and source flags. They do not establish actual contact forces, retained grasp, replay success or firm-grasp ground truth. Recorded task success is the original benchmark criterion.', '',
                   'HDF5/video paths in the new indices remain relative to dataset_root in study.json. Conditioning frames and observation-base pose transforms are rebuilt for the newly selected event.', '']
    if status['state']=='complete':
        description += ['| Split | Original inventory | Original pose labels | Early pose labels | Houses | Conditioning examples | Replaced by earlier event |', '|---|---:|---:|---:|---:|---:|---:|']
        for split, s in status['statistics'].items():
            description.append(f'| {split} | {s["source_inventory_trajectories"]} | {s["original_pose_labels"]} | {s["early_pose_labels"]} | {s["houses"]} | {s["conditioning_examples"]} | {s["labels_switched_to_earlier_event"]} |')
    display=json.loads(json.dumps(status))
    for stats in display.get('statistics',{}).values():stats.pop('by_house',None)
    description += ['', 'Machine-readable summary (full per-house counts are in status.json and house_balance.csv):', '', json.dumps(display,indent=2), '', 'Original Val inventory remains 197 trajectories / 192 original pose references. Excluded quality labels remain in the ledgers and must not be removed from a future full-task evaluation.', '']
    (run / 'report.md').write_text('\n'.join(description))
    (run / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Early pose quality view</title><style>body{font:16px system-ui;margin:40px}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(description))+'</pre><p><a href="train_early_pose_targets.jsonl">Train early labels</a> · <a href="val_early_pose_targets.jsonl">Separate Val early labels</a> · <a href="train_quality_ledger.jsonl">Train ledger</a> · <a href="val_quality_ledger.jsonl">Val ledger</a> · <a href="house_balance.csv">Per-house balance</a> · <a href="status.json">Full status</a></p>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','build','report'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--dataset',type=Path)
    parser.add_argument('--config',type=Path);parser.add_argument('--minimum-source-targets',type=int,default=0)
    parser.add_argument('--project',type=Path,default=Path('/u/xna8aw/workspace/part-aware-policy-learning'))
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='report':report(args.run)
    else:
        try:build(args.run)
        except Exception:
            write(args.run/'build_error.json',dict(error=traceback.format_exc()))
            raise


if __name__=='__main__':main()
