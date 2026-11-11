"""Select, selectively download and audit official Train expert pose references."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

import h5py
import numpy as np
import pyarrow.parquet as pq
import requests
from scipy.spatial.transform import Rotation
import zstandard

from data_preparation.data_evaluation_reservations import (check_frozen_reservations, exclude_reserved_order,
    freeze_reservations, load_reservations, require_source_disjoint)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def decode(value):
    if isinstance(value, np.ndarray):
        value = value.tobytes().rstrip(b'\0')
    return json.loads(value)


def prepare(run, config):
    study = json.loads(config.read_text())
    assert study['source_split'] == 'train'
    if (run / 'study.json').exists():
        saved = json.loads((run / 'study.json').read_text())
        assert all(saved.get(key) == value for key,value in study.items()), 'Prepared data configuration changed'
        assert digest(Path(saved['validation_root']) / 'valid_trajectory_index.json') == saved['validation_index_sha256']
        for split in ('train', 'val'):
            assert digest(run / f'metadata/{split}_pkgs.parquet') == saved['metadata_sha256'][split]
        check_frozen_reservations(run, saved, json.loads((run / 'house_order.json').read_text()))
        return
    reservations = load_reservations(Path(__file__).resolve().parents[1], study.get('reservation_registry'))
    rows = pq.read_table(run / 'metadata/train_pkgs.parquet').to_pylist()
    groups = {}
    for row in rows:
        match = re.fullmatch(r'DoorOpeningDataGenConfig_house_(\d+)\.tar\.zst', row['path'])
        if match and row['inflated_size'] > 0:
            groups.setdefault(int(match[1]), []).append(row)
    houses = sorted(groups, key=lambda h: hashlib.sha256(f"pose-data-v1:{study['seed']}:{h}".encode()).hexdigest())
    houses = [row['house'] for row in exclude_reserved_order([dict(house=h) for h in houses], reservations)]
    val_index = Path(study['validation_root']) / 'valid_trajectory_index.json'
    val = json.loads(val_index.read_text())
    # Official Val is the project's Test resource. It is allowed to contain
    # reserved Test houses; only Train selection must exclude those houses.
    study.update(validation_trajectory_count=sum(len(t) for files in val.values() for t in files.values()),
                 validation_index_sha256=digest(val_index),
                 metadata_sha256={s: digest(run / f'metadata/{s}_pkgs.parquet') for s in ['train', 'val']})
    assert study['validation_trajectory_count'] == 197
    (run / 'source').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), run / 'source/collect_interaction_pose_dataset.py')
    shutil.copy2(Path(__file__).with_name('data_evaluation_reservations.py'), run / 'source/data_evaluation_reservations.py')
    shutil.copy2(config, run / 'source/interaction_pose_dataset_v1.json')
    from data_preparation.snapshots import freeze_packages, source_hashes as snapshot_hashes
    freeze_packages(Path(__file__).resolve().parents[1], run / 'source')
    study['final_evaluation_reservations'] = freeze_reservations(run, reservations)
    write(run / 'study.json', study)
    write(run / 'house_order.json', [dict(rank=i, house=h, packages=sorted(groups[h], key=lambda r: r['part']))
                                    for i, h in enumerate(houses)])
    write(run / 'source-sha256.json', snapshot_hashes(run / 'source'))
    print(f'Prepared {len(houses)} ordered Train houses; validation has 197 preserved trajectories.', flush=True)


def fetch_package(run, study, house, entry):
    record = run / f"packages/part{entry['part']}_house_{house}.json"
    destination = run / f"raw/part{entry['part']}/train"
    target_house = destination / f'house_{house}'
    if record.exists():
        assert target_house.is_dir()
        return json.loads(record.read_text())
    # Adopt the already-downloaded first pilot package, without another transfer.
    pilot = run / f"metadata/pilot_package_part{entry['part']}_house_{house}.json"
    if pilot.exists() and target_house.exists():
        previous = json.loads(pilot.read_text())
        assert previous['entry'] == entry
        write(record, previous)
        return previous
    temporary = run / f"downloads/part{entry['part']}_house_{house}.tar.zst.partial"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    start, size = entry['offset'], entry['size']
    url = (f"https://huggingface.co/datasets/{study['repo']}/resolve/{study['revision']}/"
           f"{study['task']}/train_shards/{entry['shard_id']:05d}.tar")
    for attempt in range(3):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{start + size - 1}'},
                              stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                expected = f'bytes {start}-{start + size - 1}/'
                assert response.status_code == 206 and response.headers.get('Content-Range', '').startswith(expected), 'Server did not honor exact byte range'
                h = hashlib.sha256(); count = 0
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError('Oversized byte-range response')
                        stream.write(chunk); h.update(chunk)
                assert count == size
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    staging = run / f"downloads/unpack_part{entry['part']}_house_{house}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with temporary.open('rb') as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as archive:
                for member in archive:
                    if not (member.isfile() or member.isdir()):
                        raise ValueError('Archive links or special entries are unsupported')
                    path = (staging / member.name).resolve()
                    assert path.is_relative_to(staging.resolve()), 'Unsafe archive path'
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, path.open('wb') as output:
                            shutil.copyfileobj(source, output)
    children = list(staging.iterdir())
    assert len(children) == 1 and children[0].name == f'house_{house}'
    destination.mkdir(parents=True, exist_ok=True)
    assert not target_house.exists(), f'Untracked extraction already exists: {target_house}'
    children[0].rename(target_house)
    staging.rmdir(); temporary.unlink()
    result = dict(entry=entry, sha256=h.hexdigest(), download_bytes=count, byte_range_verified=True)
    write(record, result)
    return result


def transform(pose):
    p = np.asarray(pose, dtype=float)
    assert p.shape == (7,) and np.isfinite(p).all()
    assert abs(np.linalg.norm(p[3:]) - 1) < .02
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(p[3:], scalar_first=True).as_matrix()
    value[:3, 3] = p[:3]
    return value


def pose7(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)].tolist()


def contiguous(values):
    return [x for x in np.split(values, np.flatnonzero(np.diff(values) != 1) + 1) if len(x)]


def inspect_trajectory(run, study, house, part, h5, group, valid, preview=False):
    scene = decode(group['obs_scene'][()])
    n = len(group['success'])
    record = dict(source_split='train', part=part, house=house,
                  h5=str(h5.relative_to(run)), trajectory=group.name.strip('/'),
                  observations=n, recorded_success=bool(group['success'][:].any()), valid_trajectory_mask=valid,
                  task_description=scene.get('task_description'), policy_dt_ms=scene.get('policy_dt_ms'))
    if not valid or not record['recorded_success']:
        return record, None, 'invalid source mask or no recorded success'
    assert scene['policy_dt_ms'] == 100.0
    required = ['actions/ee_pose', 'actions/commanded_action', 'obs/agent/qpos',
                'obs/extra/left_tcp_pose', 'obs/extra/right_tcp_pose', 'obs/extra/robot_base_pose',
                'obs/extra/policy_phase', 'obs/extra/task_info', 'obs/extra/door_state']
    assert all(len(group[k]) == n for k in required)
    phase = group['obs/extra/policy_phase'][:]
    labels = scene['policy_phases']
    phase_id = int(labels['GRASP_HANDLE'])
    first_success = int(np.flatnonzero(group['success'][:])[0])
    task_info = [decode(row) for row in group['obs/extra/task_info'][:]]
    events = []
    for window in contiguous(np.flatnonzero((phase == phase_id) & (np.arange(n) <= first_success))):
        start, end = int(window[0]), int(window[-1])
        if start == 0 or len(window) < 3:
            continue
        before, after = [decode(group['obs/agent/qpos'][i]) for i in [start - 1, end]]
        closure = {side: float(np.abs(before[f'{side}_gripper']).sum() - np.abs(after[f'{side}_gripper']).sum())
                   for side in ['left', 'right']}
        side = max(closure, key=closure.get)
        stable = all(task_info[i].get('handle_grasped', False) for i in window[-3:])
        if not stable or closure[side] < study['minimum_gripper_closure_proxy_m']:
            continue
        base = transform(group['obs/extra/robot_base_pose'][end])
        tcp = transform(group[f'obs/extra/{side}_tcp_pose'][end])
        world = base @ tcp
        handle = np.asarray(decode(group['obs/extra/door_state'][end])['handle_position'])
        events.append(dict(start_frame=start, target_frame=end, arm=side, measured_tcp_base_xyz_wxyz=pose7(tcp),
                           measured_tcp_world_xyz_wxyz=pose7(world), robot_base_world_xyz_wxyz=pose7(base),
                           gripper_joint_positions=after[f'{side}_gripper'], closure_proxy_m=closure[side],
                           handle_position_world=handle.tolist(), tcp_minus_handle_world_m=(world[:3, 3] - handle).tolist(),
                           tcp_to_handle_origin_m=float(np.linalg.norm(world[:3, 3] - handle)),
                           commanded_ee_pose=decode(group['actions/ee_pose'][end]).get(f'{side}_gripper')))
    record.update(first_success_frame=first_success, grasp_event_count=len(events))
    if not events:
        return record, None, 'no grasp-phase event with stable grasp flag and clear closure'
    event = events[-1]
    start = event['start_frame']
    approach = np.flatnonzero((phase == int(labels['REACH_PRE_GRASP'])) & (np.arange(n) < start))
    approach_start = int(contiguous(approach)[-1][0]) if len(approach) else max(0, start - 20)
    condition_frames = sorted(set(np.linspace(approach_start, start - 1, 4, dtype=int).tolist()))
    cameras = {}
    for name in ['head_camera', 'wrist_camera_l', 'wrist_camera_r']:
        relative = group[f'obs/sensor_data/{name}'][:].tobytes().rstrip(b'\0').decode()
        video = h5.parent / relative
        assert video.is_file(), video
        cameras[name] = str(video.relative_to(run))
        from decord import VideoReader
        reader = VideoReader(str(video), num_threads=1)
        assert len(reader) == n, (name, len(reader), n)
        if preview:
            for frame in [condition_frames[0], event['target_frame'], first_success]:
                assert reader[frame].shape == (576, 1024, 3)
    conditions = []
    target_world = transform(event['measured_tcp_world_xyz_wxyz'])
    for i in condition_frames:
        observation_base = transform(group['obs/extra/robot_base_pose'][i])
        conditions.append(dict(frame=i, target_tcp_in_observation_base_xyz_wxyz=pose7(np.linalg.inv(observation_base) @ target_world)))
    target = dict(**record, event_definition=study['label_rule'], interaction_event=event,
                  other_eligible_events=events[:-1], condition_frames=conditions, videos=cameras,
                  pose_convention='xyz, quaternion wxyz; measured TCP is robot-base relative',
                  part_relative_label_status='World translation relative to handle origin only; handle rotation is unavailable offline',
                  replay_status='Not replay validated; old scene/robot asset compatibility remains to be checked',
                  video_observation_counts_checked=True)
    if preview:
        from decord import VideoReader
        from PIL import Image, ImageDraw
        reader = VideoReader(str(run / cameras['head_camera']), num_threads=1)
        frames = [condition_frames[0], start, event['target_frame'], first_success]
        sheet = Image.new('RGB', (1280, 230), '#132638')
        draw = ImageDraw.Draw(sheet)
        for j, (i, label) in enumerate(zip(frames, ['Approach', 'Grasp phase starts', 'Target measured pose', 'Task success'])):
            sheet.paste(Image.fromarray(reader[i].asnumpy()).resize((320, 180)), (320 * j, 36))
            draw.text((320 * j + 8, 8), f'{label}: frame {i}', fill='white')
        path = run / f"audit/previews/part{part}_house_{house}_{record['trajectory']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path, quality=90)
        target['audit_preview'] = str(path.relative_to(run))
    return record, target, None


def publish(run, study, houses, state):
    records = [r for h in houses for r in h['records']]
    targets = [t for h in houses for t in h['targets']]
    for name, values in [('train_trajectories.jsonl', records), ('pose_targets.jsonl', targets)]:
        path = run / name; temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(v, allow_nan=False) + '\n' for v in values)); temporary.replace(path)
    assert digest(Path(study['validation_root']) / 'valid_trajectory_index.json') == study['validation_index_sha256']
    result = dict(state=state, houses=len(houses), downloaded_packages=sum(h['packages'] for h in houses),
                  download_bytes=sum(h['download_bytes'] for h in houses), source_trajectories=len(records),
                  successful_valid_trajectories=sum(r['valid_trajectory_mask'] and r['recorded_success'] for r in records),
                  pose_trajectories=len(targets), pose_condition_examples=sum(len(t['condition_frames']) for t in targets),
                  initial_target=study['initial_target_pose_trajectories'],
                  pilot_audited=sum('audit_preview' in t for t in targets),
                  tasks=dict(Counter(t['task_description'] for t in targets)),
                  active_arms=dict(Counter(t['interaction_event']['arm'] for t in targets)),
                  exclusions=dict(Counter(reason for h in houses for reason in h['exclusions'])),
                  validation_trajectories=197, validation_unchanged=True,
                  label_status=study['label_status'], trained_model=False)
    write(run / 'status.json', result)
    from html import escape
    previews = ''.join(f'<figure><img style="width:100%" src="{t["audit_preview"]}"><figcaption>House {t["house"]}, {t["trajectory"]}: {escape(t["task_description"])}; {t["interaction_event"]["arm"]} arm</figcaption></figure>'
                       for t in targets if 'audit_preview' in t)
    (run / 'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>Interaction-pose expert subset</title><style>body{{font:16px system-ui;max-width:1400px;margin:30px auto}}pre{{white-space:pre-wrap}}</style>
<h1>Official Train expert subset for interaction-pose learning</h1><p>Offline label audit and data preparation. Original validation data remains separate. Poses are expert references; native replay and part-frame reconstruction are pending.</p>
<pre>{escape(json.dumps(result,indent=2))}</pre><p><a href="study.json">Protocol</a> · <a href="pose_targets.jsonl">Pose references</a> · <a href="train_trajectories.jsonl">Trajectory inventory</a></p>{previews}''')
    return result


def index_validation(run, study):
    """Index the released expert Test source without copying or changing it."""
    if (run / 'validation_status.json').exists():
        if digest(Path(study['validation_root']) / 'valid_trajectory_index.json') != study['validation_index_sha256']:
            raise ValueError('Released Val source inventory changed')
        return
    source = Path(study['validation_root'])
    reference = run / 'validation_reference'
    if not reference.exists():
        reference.symlink_to(source, target_is_directory=True)
    if reference.resolve() != source.resolve():
        raise ValueError('Validation source link points to a different dataset')
    inventory = json.loads((source / 'valid_trajectory_index.json').read_text())
    records, targets = [], []
    for house_name, files in sorted(inventory.items()):
        house = int(house_name.removeprefix('house_'))
        for relative, trajectories in sorted(files.items()):
            h5 = reference / relative
            file_hash = digest(h5)
            with h5py.File(h5, 'r') as data:
                for name in sorted(trajectories):
                    record, target, reason = inspect_trajectory(run, study, house, 0, h5, data[name], True)
                    record.update(source_split='val', h5_sha256=file_hash, pose_eligible=target is not None)
                    if reason:
                        record['pose_exclusion'] = reason
                    records.append(record)
                    if target is not None:
                        target.update(source_split='val', h5_sha256=file_hash)
                        targets.append(target)
    for name, rows in [('val_trajectories.jsonl', records), ('val_pose_targets.jsonl', targets)]:
        (run / name).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    if len(records) != study['validation_trajectory_count']:
        raise ValueError('Released Val trajectory inventory changed during extraction')
    write(run / 'validation_status.json', dict(state='complete', source_split='val',
        trajectories=len(records), pose_trajectories=len(targets), raw_source_read_only=True,
        physical_grasp_retention_certified=False))


def collect(run, pilot_only=False):
    study = json.loads((run / 'study.json').read_text())
    index_validation(run, study)
    target_count = study['pilot_target_pose_trajectories'] if pilot_only else study['initial_target_pose_trajectories']
    order = json.loads((run / 'house_order.json').read_text())
    check_frozen_reservations(run, study, order)
    houses = []
    for item in order:
        checkpoint = run / f"houses/house_{item['house']}.json"
        if checkpoint.exists():
            houses.append(json.loads(checkpoint.read_text()))
        else:
            previous = publish(run, study, houses, 'collecting')
            if previous['pose_trajectories'] >= target_count:
                break
            assert len(houses) < study['maximum_houses'], 'House limit reached; review audit before expanding'
            estimate = previous['download_bytes'] + sum(p['size'] for p in item['packages'])
            assert estimate <= study['download_budget_bytes'], 'Download budget reached'
            print(f"Downloading house {item['house']} (rank {item['rank']})", flush=True)
            house = dict(house=item['house'], rank=item['rank'], packages=0, download_bytes=0, records=[], targets=[], exclusions=[])
            audited = previous['pilot_audited']
            for entry in item['packages']:
                package = fetch_package(run, study, item['house'], entry)
                house['packages'] += 1; house['download_bytes'] += package['download_bytes']
                directory = run / f"raw/part{entry['part']}/train/house_{item['house']}"
                for h5 in sorted(directory.glob('*.h5')):
                    h5_hash = digest(h5)
                    with h5py.File(h5, 'r') as data:
                        names = sorted((k for k in data if k.startswith('traj_')), key=lambda k: int(k.split('_')[-1]))
                        mask = data['valid_traj_mask'][:] if 'valid_traj_mask' in data else None
                        for name in names:
                            i = int(name.split('_')[-1]); valid = bool(mask[i]) if mask is not None else True
                            try:
                                record, target, reason = inspect_trajectory(run, study, item['house'], entry['part'], h5, data[name], valid,
                                                                          preview=audited < study['pilot_target_pose_trajectories'])
                                record['h5_sha256'] = h5_hash
                                record['pose_eligible'] = target is not None
                                if reason:
                                    record['pose_exclusion'] = reason
                                house['records'].append(record)
                                if target:
                                    target['h5_sha256'] = h5_hash; house['targets'].append(target)
                                    audited += 'audit_preview' in target
                                if reason:
                                    house['exclusions'].append(reason)
                            except Exception as error:
                                house['records'].append(dict(h5=str(h5.relative_to(run)), trajectory=name, source_split='train',
                                                             valid_trajectory_mask=False, recorded_success=False, error=str(error)))
                                house['exclusions'].append(type(error).__name__ + ': ' + str(error))
            write(checkpoint, house); houses.append(house)
        result = publish(run, study, houses, 'collecting')
        print(json.dumps(result), flush=True)
        if result['pose_trajectories'] >= target_count:
            break
    result = publish(run, study, houses, 'pilot_complete' if pilot_only else 'complete')
    assert result['pose_trajectories'] >= target_count
    assert result['pilot_audited'] >= study['pilot_target_pose_trajectories']
    print('COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--metadata-dir', type=Path, help='Pinned train_pkgs.parquet and val_pkgs.parquet inputs')
    args = parser.parse_args()
    if args.prepare:
        if args.config is None:
            parser.error('--prepare requires --config')
        for folder in ('metadata', 'raw', 'packages', 'houses', 'downloads', 'audit'):
            (args.run / folder).mkdir(parents=True, exist_ok=True)
        if args.metadata_dir is not None:
            for split in ('train', 'val'):
                original = args.metadata_dir / f'{split}_pkgs.parquet'
                target = args.run / 'metadata' / original.name
                if target.exists() and digest(target) != digest(original):
                    raise ValueError('Pinned metadata changed: ' + split)
                if not target.exists():
                    shutil.copy2(original, target)
        prepare(args.run.resolve(), args.config.resolve())
    else:
        try:
            collect(args.run.resolve(), args.pilot_only)
        except Exception as error:
            write(args.run / 'collection_error.json', dict(error=repr(error), type=type(error).__name__))
            raise
