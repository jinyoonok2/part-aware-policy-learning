"""Verify split isolation, source integrity and measured-pose transforms."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


def main(run):
    from data_preparation.snapshots import load_source_module
    extraction = load_source_module('frozen_pose_extractor', run / 'source/collect_interaction_pose_dataset.py')
    study = json.loads((run / 'study.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'complete'
    by_split = {}
    hashes = {}
    conditions = 0
    for split, filename in [('train', 'pose_targets.jsonl'), ('val', 'val_pose_targets.jsonl')]:
        rows = [json.loads(line) for line in (run / filename).read_text().splitlines()]
        by_split[split] = rows
        grouped = defaultdict(list)
        identities = set()
        for row in rows:
            assert row['source_split'] == split and row['valid_trajectory_mask'] and row['recorded_success']
            path = (run / row['h5']).resolve()
            expected_root = (run / 'raw').resolve() if split == 'train' else Path(study['validation_root']).resolve()
            assert path.is_relative_to(expected_root)
            identity = (row['h5_sha256'], row['trajectory'])
            assert identity not in identities, f'Duplicate source trajectory: {identity}'
            identities.add(identity)
            grouped[path].append(row)
        for path, values in grouped.items():
            hashes[path] = extraction.digest(path)
            with h5py.File(path, 'r') as data:
                for row in values:
                    assert hashes[path] == row['h5_sha256']
                    group = data[row['trajectory']]
                    event = row['interaction_event']; frame = event['target_frame']; side = event['arm']
                    actual = extraction.transform(group['obs/extra/robot_base_pose'][frame]) @ extraction.transform(group[f'obs/extra/{side}_tcp_pose'][frame])
                    saved = extraction.transform(event['measured_tcp_world_xyz_wxyz'])
                    assert np.allclose(actual, saved, atol=1e-6, rtol=0)
                    qpos = extraction.decode(group['obs/agent/qpos'][frame])
                    assert np.allclose(qpos[f'{side}_gripper'], event['gripper_joint_positions'])
                    assert frame <= row['first_success_frame']
                    assert all(extraction.decode(group['obs/extra/task_info'][j])['handle_grasped'] for j in range(frame - 2, frame + 1))
                    for condition in row['condition_frames']:
                        i = condition['frame']; assert 0 <= i < event['start_frame']
                        restored = extraction.transform(group['obs/extra/robot_base_pose'][i]) @ extraction.transform(condition['target_tcp_in_observation_base_xyz_wxyz'])
                        assert np.allclose(restored, saved, atol=1e-6, rtol=0)
                        conditions += 1
                    assert all((run / value).is_file() for value in row['videos'].values())
    train_sources = {(r['h5_sha256'], r['trajectory']) for r in by_split['train']}
    val_sources = {(r['h5_sha256'], r['trajectory']) for r in by_split['val']}
    assert train_sources.isdisjoint(val_sources)
    assert len(by_split['train']) >= study['initial_target_pose_trajectories']
    assert len(by_split['val']) == json.loads((run / 'validation_status.json').read_text())['pose_trajectories']
    assert extraction.digest(Path(study['validation_root']) / 'valid_trajectory_index.json') == study['validation_index_sha256']
    source_hashes = json.loads((run / 'source-sha256.json').read_text())
    for name, expected in source_hashes.items():
        assert extraction.digest(run / 'source' / name) == expected
    result = dict(state='passed', train_pose_trajectories=len(by_split['train']), val_pose_trajectories=len(by_split['val']),
                  conditioning_transforms_checked=conditions, source_hdf5_hashes_checked=len(hashes),
                  duplicate_source_trajectories=0, train_val_source_overlap=0, validation_index_unchanged=True,
                  position_transform_tolerance_m=1e-6, physical_replay_validated=False, part_masks_generated=False,
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    extraction.write(run / 'audit/dataset_integrity.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
