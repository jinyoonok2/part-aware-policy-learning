"""Prepare two project splits: official Train for learning, official Val as Test.

Old data stay read-only. The released Val catalog is completed, not resampled
for policy outcomes. Pose-reference eligibility is separate from physical
grasp-retention feasibility, which this CPU preparation does not establish.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import traceback


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def write_lines(path, rows):
    path = Path(path); temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    temporary.replace(path)


def case_key(row):
    return int(row['house']), int(row.get('part', 0)), Path(row['h5']).name, row['trajectory']


def source_key(row):
    return row.get('h5_sha256'), row.get('trajectory')


REQUIRED_STATE_ARRAYS = ('actions/ee_pose', 'actions/commanded_action', 'obs/agent/qpos',
    'obs/extra/left_tcp_pose', 'obs/extra/right_tcp_pose', 'obs/extra/robot_base_pose',
    'obs/extra/policy_phase', 'obs/extra/task_info', 'obs/extra/door_state')
RGB_CAMERAS = ('head_camera', 'wrist_camera_l', 'wrist_camera_r')


def validate_trajectory_alignment(group, h5_path, run, video_reader=None):
    """Check every source case, including those ineligible for pose labels."""
    context = f'{h5_path}:{getattr(group, "name", "trajectory")}'
    if 'success' not in group or len(group['success']) == 0:
        raise ValueError('Missing/empty observations: ' + context)
    count = len(group['success'])
    for name in REQUIRED_STATE_ARRAYS:
        if name not in group or len(group[name]) != count:
            raise ValueError('Missing or misaligned state/action array ' + name + ': ' + context)
    if video_reader is None:
        from decord import VideoReader
        video_reader = VideoReader
    videos = {}
    for camera in RGB_CAMERAS:
        name = 'obs/sensor_data/' + camera
        if name not in group:
            raise ValueError('Missing RGB video annotation ' + camera + ': ' + context)
        raw = group[name][:]
        relative = (raw if isinstance(raw, bytes) else raw.tobytes()).rstrip(b'\0').decode('utf8')
        path = Path(h5_path).parent / relative
        if not path.is_file() or not path.resolve().is_relative_to(Path(h5_path).parent.resolve()):
            raise ValueError('Missing or external RGB video ' + camera + ': ' + context)
        if len(video_reader(str(path), num_threads=1)) != count:
            raise ValueError('RGB video/observation count mismatch ' + camera + ': ' + context)
        videos[camera] = str(path.relative_to(run))
    return dict(observations=count, videos=videos, video_observation_counts_checked=True,
                required_state_observation_counts_checked=True)


def with_role(row, role, root, exposure=None):
    expected = {'train': 'train', 'test': 'val'}[role]
    if row.get('source_split') != expected:
        raise ValueError(f'{role} alias must retain official source_split={expected}')
    result = dict(row, project_role=role, dataset_root=str(root))
    if role == 'test':
        reason = (exposure or {}).get(str(row['house']))
        result.update(previous_project_exposure=bool(reason), exposure_note=reason,
                      stable_grasp_feasibility_verified=False)
    return result


def require_disjoint(train, test, extra_train_houses=()):
    train_houses = {int(row['house']) for row in train} | set(extra_train_houses)
    test_houses = {int(row['house']) for row in test}
    overlap = train_houses & test_houses
    if overlap:
        raise ValueError(f'Train/Test numeric house overlap: {sorted(overlap)}')
    # Hash checks are independent of a path spelling or symlink location.
    train_files = {row['h5_sha256'] for row in train if row.get('h5_sha256')}
    test_files = {row['h5_sha256'] for row in test if row.get('h5_sha256')}
    if train_files & test_files:
        raise ValueError('Train/Test raw HDF5 hash overlap')
    train_ids = {source_key(row) for row in train if row.get('h5_sha256')}
    test_ids = {source_key(row) for row in test if row.get('h5_sha256')}
    if train_ids & test_ids:
        raise ValueError('Train/Test trajectory identity overlap')


def safe_members(archive, destination, maximum_bytes):
    """Extract a private staging tree; reject traversal, links and special files."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('Extraction root must not be a symlink')
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(); total = 0; names = []
    for member in archive:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in member.name:
            raise ValueError('Unsafe archive path: ' + member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError('Archive links/special files are unsupported: ' + member.name)
        path = destination.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError('Archive path escapes extraction root')
        if path.exists() and not (member.isdir() and path.is_dir()):
            raise ValueError('Archive would overwrite an existing file')
        if member.isdir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            total += member.size
            if total > maximum_bytes:
                raise ValueError('Archive exceeds its declared inflated byte budget')
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('xb') as output:
                shutil.copyfileobj(source, output)
        names.append(member.name)
    return dict(member_names=names, unpacked_file_bytes=total)


def load_module(name, path):
    from data_preparation.snapshots import load_source_module
    return load_source_module(name, path)


def check_frozen(run, study):
    for relative, expected in study['source_hashes'].items():
        if digest(run / 'source' / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for path, expected in study['parent_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Original read-only source changed: ' + path)
    if digest(run / 'metadata/val_pkgs.parquet') != study['catalog_sha256']:
        raise ValueError('Frozen Val package catalog changed')


def prepare(args):
    import pyarrow.parquet as pq
    config = json.loads(args.config.read_text())
    parent = Path(config.get('source_dataset', config.get('parent_dataset', ''))).resolve()
    quality = Path(config['quality_view']).resolve()
    original = json.loads((parent / 'study.json').read_text())
    if config.get('revision', config.get('source_revision')) != original['revision']:
        raise ValueError('Test source revision must match the frozen Train source revision')
    if config.get('repo', original['repo']) != original['repo'] or config.get('task', original['task']) != original['task']:
        raise ValueError('Test catalog must use the frozen official task and repository')
    if digest(parent / 'metadata/val_pkgs.parquet') != original['metadata_sha256']['val']:
        raise ValueError('Original Val package catalog identity changed')
    quality_original = json.loads((quality / 'study.json').read_text())
    if Path(quality_original['dataset_root']).resolve() != parent or quality_original['dataset_study_sha256'] != digest(parent / 'study.json'):
        raise ValueError('Early-pose view belongs to a different source dataset')
    if json.loads((parent / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Original data collection is incomplete')
    if json.loads((quality / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Original early-pose view is incomplete')
    if args.run.exists():
        study = json.loads((args.run / 'study.json').read_text())
        if study['configuration'] != config:
            raise ValueError('Resume configuration changed')
        check_frozen(args.run, study)
        return
    train = read_lines(parent / 'train_trajectories.jsonl')
    previous_val = read_lines(parent / 'val_trajectories.jsonl')
    train_houses = {int(p.name.removeprefix('house_')) for p in parent.glob('raw/part*/train/house_*')}
    source_root = Path(original['validation_root']).resolve()
    catalog = pq.read_table(parent / 'metadata/val_pkgs.parquet').to_pylist()
    rows = []
    for entry in catalog:
        match = re.fullmatch(r'DoorOpeningDataGenConfig_house_(\d+)\.tar\.zst', entry['path'])
        if not match: continue
        if int(entry.get('part', 0)) != 0:
            raise ValueError('Additional Val parts require an explicitly supported local source layout')
        house = int(match[1]); existing = source_root / f'house_{house}'
        reusable = existing.is_dir() and any(existing.glob('*.h5'))
        rows.append(dict(house=house, part=0, entry=entry, reuse_source=str(existing) if reusable else None,
                         action='reuse_read_only_house' if reusable else 'fetch_released_package'))
    if len({row['house'] for row in rows}) != len(rows):
        raise ValueError('Ambiguous Val catalog house identity')
    rows.sort(key=lambda row: (row['house'], row['part']))
    require_disjoint(train, rows, train_houses)
    transfer = sum(row['entry']['size'] for row in rows if not row['reuse_source'])
    if transfer > config.get('download_budget_bytes', 20_000_000_000):
        raise ValueError('Missing Val package transfers exceed the configured budget')
    for folder in ('source', 'metadata', 'raw/part0/val', 'packages', 'downloads', 'houses', 'logs', 'audit'):
        (args.run / folder).mkdir(parents=True, exist_ok=False)
    parent_files = [parent / name for name in ('study.json', 'status.json', 'train_trajectories.jsonl',
        'pose_targets.jsonl', 'val_trajectories.jsonl', 'val_pose_targets.jsonl', 'source-sha256.json')]
    parent_files += [quality / name for name in ('study.json', 'status.json', 'train_early_pose_targets.jsonl',
        'val_early_pose_targets.jsonl')]
    parent_files += [source_root / 'valid_trajectory_index.json']
    parent_hashes = {str(path): digest(path) for path in parent_files}
    extractor = parent / 'source/collect_interaction_pose_dataset.py'
    original_source = json.loads((parent / 'source-sha256.json').read_text())
    if digest(extractor) != original_source['collect_interaction_pose_dataset.py']:
        raise ValueError('Original pose extractor changed')
    quality_study = json.loads((quality / 'study.json').read_text())
    early = quality / 'source/build_early_interaction_pose_view.py'
    if digest(early) != quality_study['frozen_source_sha256'][early.name]:
        raise ValueError('Original early-pose extractor changed')
    shutil.copy2(extractor, args.run / 'source/frozen_source_extractor.py')
    shutil.copy2(early, args.run / 'source/frozen_early_extractor.py')
    for name in ('data_preparation/prepare_train_test_dataset.py', 'data_preparation/test_train_test_dataset.py',
                 'data_preparation/slurm/prepare_train_test_dataset.sh'):
        shutil.copy2(args.project / name, args.run / 'source' / Path(name).name)
    shutil.copy2(args.config, args.run / 'source/train_test_config.json')
    shutil.copy2(parent / 'metadata/val_pkgs.parquet', args.run / 'metadata/val_pkgs.parquet')
    from data_preparation.snapshots import freeze_packages, source_hashes as snapshot_hashes
    freeze_packages(args.project, args.run / 'source')
    study = dict(name=config.get('name', 'Official Train / official Val-as-Test'), configuration=config,
        source_dataset=str(parent), quality_view=str(quality), validation_root=str(source_root),
        project_roles={'train':'Official Train for learning', 'test':'Official Val for project testing'},
        repo=config.get('repo',original['repo']), revision=original['revision'], task=config.get('task',original['task']),
        package_order=rows, catalog_sha256=digest(args.run / 'metadata/val_pkgs.parquet'),
        catalog_house_count=len(rows), train_house_ids=sorted(train_houses | {r['house'] for r in train}),
        existing_test_trajectories=len(previous_val), estimated_new_transfer_bytes=transfer,
        parent_hashes=parent_hashes, source_hashes=snapshot_hashes(args.run / 'source'),
        selection='Entire released Val catalog, no filtering by model outcomes; prior exposure is flagged, not silently removed.',
        physical_grasp_retention_certified=False, training_started=False, model_evaluation_started=False)
    write(args.run / 'study.json', study)
    write(args.run / 'status.json', dict(state='prepared', catalog_houses=len(rows),
        original_test_trajectories=len(previous_val), estimated_new_transfer_bytes=transfer,
        physical_grasp_retention_certified=False, training_started=False, model_evaluation_started=False))
    report(args)


def fetch_package(run, study, row):
    import requests
    import zstandard
    house = row['house']; entry = row['entry']; receipt = run / 'packages' / f'house_{house}.json'
    destination = run / 'raw/part0/val' / f'house_{house}'
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value['entry'] != entry or not destination.is_dir():
            raise ValueError('Previously fetched package identity changed')
        return value
    if destination.exists() or destination.is_symlink():
        raise ValueError('Refusing to overwrite an unreceipted Test house')
    temporary = run / 'downloads' / f'house_{house}.tar.zst.partial'
    start, size = int(entry['offset']), int(entry['size'])
    url = (f"https://huggingface.co/datasets/{study['repo']}/resolve/{study['revision']}/"
           f"{study['task']}/val_shards/{int(entry['shard_id']):05d}.tar")
    for attempt in range(3):
        try:
            with requests.get(url, headers={'Range':f'bytes={start}-{start+size-1}'}, stream=True,
                              timeout=(20,120)) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{start+size-1}/'):
                    raise ValueError('Server did not honor the exact released package byte range')
                count = 0
                with temporary.open('wb') as output:
                    for block in response.iter_content(1024 * 1024):
                        count += len(block)
                        if count > size: raise ValueError('Oversized package response')
                        output.write(block)
                if count != size: raise ValueError('Incomplete package response')
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(2)
    compressed_sha = digest(temporary)
    staging = run / 'downloads' / f'unpack_house_{house}'
    if staging.exists():
        if staging.is_symlink(): raise ValueError('Unsafe existing staging symlink')
        shutil.rmtree(staging)
    staging.mkdir()
    with temporary.open('rb') as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream,mode='r|') as archive:
                extracted = safe_members(archive, staging, int(entry['inflated_size']))
    children = list(staging.iterdir())
    if len(children) != 1 or children[0].name != f'house_{house}' or not children[0].is_dir():
        raise ValueError('Released package root differs from its catalog house')
    children[0].rename(destination); staging.rmdir(); temporary.unlink()
    result = dict(entry=entry, compressed_sha256=compressed_sha, exact_byte_range_verified=True,
                  downloaded_bytes=size, source_url=url, **extracted)
    write(receipt,result)
    return result


def rebuild_early(target, group, extraction, early, source_study, quality_study):
    import numpy as np
    scene = extraction.decode(group['obs_scene'][()])
    fractions = [float(np.asarray(extraction.decode(x)['opening_percentage']).reshape(-1)[0])
                 for x in group['obs/extra/door_state']]
    events = target.get('other_eligible_events', []) + [target['interaction_event']]
    selected, _ = early.select_event(events, fractions, quality_study['maximum_source_opening_fraction'],
                                     quality_study['require_no_prior_major_opening'])
    if selected is None: return None, 'no eligible initial grasp before prior major opening'
    world = early.source_event(group,scene,selected,extraction,source_study,quality_study['minimum_grasp_phase_samples'])
    frames, _ = early.conditioning_frames(group['obs/extra/policy_phase'][:],
        int(scene['policy_phases']['REACH_PRE_GRASP']), selected['start_frame'], quality_study['conditioning_samples'])
    conditions = early.conditioning_records(group['obs/extra/robot_base_pose'][:],world,frames,fractions)
    for condition in conditions:
        actual = extraction.transform(group['obs/extra/robot_base_pose'][condition['frame']]) @ extraction.transform(condition['target_tcp_in_observation_base_xyz_wxyz'])
        if not np.allclose(actual,world,atol=1e-6,rtol=0): raise ValueError('Test target frame transform failed')
    result = dict(target, interaction_event=selected, other_eligible_events=[e for e in events if e is not selected],
        condition_frames=conditions,event_definition=quality_study['label_definition'],
        original_target_frame=target['interaction_event']['target_frame'],
        original_event_definition=target['event_definition'],selected_source_opening_fraction=fractions[selected['target_frame']],
        selected_source_prior_max_opening_fraction=max(fractions[:selected['target_frame']+1]),
        quality_label_confidence=quality_study['label_confidence'],quality_view='initial_grasp_before_major_opening_v1')
    return result, None


def validate_reference_preservation(new, previous, kind):
    import numpy as np
    lookup = {case_key(row):row for row in new}
    if len(lookup) != len(new): raise ValueError('Duplicate Test trajectory identity')
    for row in previous:
        actual = lookup.get(case_key(row))
        if actual is None: raise ValueError('An existing '+kind+' reference was lost')
        if actual['h5_sha256'] != row['h5_sha256']:
            raise ValueError('Preserved Test source HDF5 changed')
        if kind != 'trajectory':
            if actual['interaction_event']['target_frame'] != row['interaction_event']['target_frame']:
                raise ValueError('Existing '+kind+' event selection changed')
            if not np.allclose(actual['interaction_event']['measured_tcp_world_xyz_wxyz'],
                               row['interaction_event']['measured_tcp_world_xyz_wxyz'],atol=1e-6,rtol=0):
                raise ValueError('Existing '+kind+' target changed')


def collect(args):
    import fcntl
    import h5py
    run=args.run;study=json.loads((run/'study.json').read_text());check_frozen(run,study)
    lock=(run/'.collection.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=Path(study['source_dataset']);quality=Path(study['quality_view'])
    source_study=json.loads((parent/'study.json').read_text())
    quality_study=json.loads((quality/'study.json').read_text())
    extraction=load_module('train_test_frozen_source',run/'source/frozen_source_extractor.py')
    early=load_module('train_test_frozen_early',run/'source/frozen_early_extractor.py')
    exposure=study['configuration'].get('exposure_houses',{})
    valid_index=json.loads((Path(study['validation_root'])/'valid_trajectory_index.json').read_text())
    houses=[]
    for row in study['package_order']:
        house=row['house'];checkpoint=run/'houses'/f'house_{house}.json'
        if checkpoint.exists():
            value=json.loads(checkpoint.read_text())
            for relative,sha in value['h5_sha256'].items():
                if digest(run/relative)!=sha:raise ValueError('Read-only Test HDF5 changed')
            houses.append(value);continue
        write(run/'status.json',dict(state='collecting',completed_catalog_houses=len(houses),current_house=house,
            training_started=False,model_evaluation_started=False,physical_grasp_retention_certified=False))
        folder=run/'raw/part0/val'/f'house_{house}'
        if row['reuse_source']:
            if folder.is_symlink():
                if folder.resolve()!=Path(row['reuse_source']).resolve():raise ValueError('Reused house link changed')
            elif folder.exists():raise ValueError('Reused Test house has an unexpected local directory')
            else:folder.symlink_to(Path(row['reuse_source']),target_is_directory=True)
        else:fetch_package(run,study,row)
        records=[];targets=[];early_targets=[];hashes={}
        for path in sorted(folder.glob('*.h5')):
            sha=digest(path);hashes[str(path.relative_to(run))]=sha
            with h5py.File(path,'r') as data:
                names=sorted((name for name in data if name.startswith('traj_')),key=lambda name:int(name.split('_')[-1]))
                mask=data['valid_traj_mask'][:] if 'valid_traj_mask' in data else None
                for name in names:
                    alignment=validate_trajectory_alignment(data[name],path,run)
                    if mask is not None:valid=bool(mask[int(name.split('_')[-1])])
                    else:valid=name in valid_index.get(f'house_{house}',{}).get(f'house_{house}/{path.name}',{})
                    record,target,reason=extraction.inspect_trajectory(run,source_study,house,row['part'],path,data[name],valid,preview=False)
                    record.update(source_split='val',h5_sha256=sha,pose_exclusion=reason)
                    record.update(alignment)
                    record=with_role(record,'test',run,exposure)
                    record.update(general_pose_eligible=target is not None,early_pose_eligible=False)
                    if target is not None:
                        target.update(source_split='val',h5_sha256=sha,required_state_observation_counts_checked=True)
                        target=with_role(target,'test',run,exposure);targets.append(target)
                        initial,early_reason=rebuild_early(target,data[name],extraction,early,source_study,quality_study)
                        if initial is not None:early_targets.append(initial);record['early_pose_eligible']=True
                        record['early_pose_exclusion']=early_reason
                    else:record['early_pose_exclusion']=reason
                    records.append(record)
            if digest(path)!=sha:raise ValueError('Source HDF5 changed during read-only extraction')
        value=dict(house=house,part=row['part'],source_split='val',project_role='test',records=records,
            general_targets=targets,early_targets=early_targets,h5_sha256=hashes,
            package_has_hdf5=bool(hashes),source_status='indexed' if hashes else 'released_package_has_no_trajectory_hdf5')
        write(checkpoint,value);houses.append(value)
    records=[r for h in houses for r in h['records']];targets=[r for h in houses for r in h['general_targets']]
    early_targets=[r for h in houses for r in h['early_targets']]
    records.sort(key=case_key);targets.sort(key=case_key);early_targets.sort(key=case_key)
    old_records=read_lines(parent/'val_trajectories.jsonl')
    validate_reference_preservation(records,old_records,'trajectory')
    validate_reference_preservation(targets,read_lines(parent/'val_pose_targets.jsonl'),'general pose')
    validate_reference_preservation(early_targets,read_lines(quality/'val_early_pose_targets.jsonl'),'early pose')
    train=[with_role(r,'train',parent) for r in read_lines(parent/'train_trajectories.jsonl')]
    train_targets=[with_role(r,'train',parent) for r in read_lines(parent/'pose_targets.jsonl')]
    train_early=[with_role(r,'train',parent) for r in read_lines(quality/'train_early_pose_targets.jsonl')]
    require_disjoint(train,records,study['train_house_ids'])
    for name,rows in [('train_trajectories.jsonl',train),('train_pose_targets.jsonl',train_targets),
        ('train_early_pose_targets.jsonl',train_early),('test_trajectories.jsonl',records),
        ('test_pose_targets.jsonl',targets),('test_early_pose_targets.jsonl',early_targets)]:
        write_lines(run/name,rows)
    cases=[dict(case_id=f'val_house_{r["house"]}_part{r["part"]}_{Path(r["h5"]).stem}_{r["trajectory"]}',
        house=r['house'],part=r['part'],source_split='val',project_role='test',h5=r['h5'],trajectory=r['trajectory'],
        h5_sha256=r['h5_sha256'],dataset_root=str(run),general_pose_eligible=r['general_pose_eligible'],
        early_pose_eligible=r['early_pose_eligible'],previous_project_exposure=r['previous_project_exposure'],
        exposure_note=r['exposure_note'],stable_grasp_feasibility_verified=False,final_eligible=False) for r in records]
    write(run/'final_episode_manifest.json',dict(project_split='Train / Test only',official_test_source='val',
        cases=cases,house_ids=sorted({r['house'] for r in records}),catalog_house_ids=[r['house'] for r in study['package_order']],
        physical_grasp_retention_certified=False,model_evaluation_ready=False,
        reason='Source-backed Test inventory; compatible scene staging and expert stable-grasp witnesses remain separate work.'))
    check_frozen(run,study)
    counts=dict(train_trajectories=len(train),train_general_pose_references=len(train_targets),
        train_early_pose_references=len(train_early),
        train_processed_source_houses=json.loads((parent/'status.json').read_text())['houses'],
        train_inventory_houses=len({r['house'] for r in train}),
        train_general_pose_houses=len({r['house'] for r in train_targets}),
        train_early_pose_houses=len({r['house'] for r in train_early}),
        test_trajectories=len(records),test_general_pose_references=len(targets),
        test_early_pose_references=len(early_targets),test_conditioning_examples=sum(len(r['condition_frames']) for r in early_targets),
        test_houses=len({r['house'] for r in records}),released_val_catalog_houses=len(houses),
        test_general_pose_houses=len({r['house'] for r in targets}),
        test_early_pose_houses=len({r['house'] for r in early_targets}),
        catalog_houses_without_hdf5=[h['house'] for h in houses if not h['package_has_hdf5']],
        added_test_trajectories=len(records)-len(old_records),previously_exposed_test_trajectories=sum(r['previous_project_exposure'] for r in records),
        test_tasks=dict(Counter(r.get('task_description') for r in records)),
        general_pose_exclusions=dict(Counter(r['pose_exclusion'] for r in records if not r['general_pose_eligible'])),
        early_pose_exclusions=dict(Counter(r['early_pose_exclusion'] for r in records if not r['early_pose_eligible'])))
    manifests={p.name:digest(p) for p in run.glob('*_*.jsonl')}
    write(run/'audit/integrity.json',dict(state='passed',original_manifests_unchanged=True,existing_test_cases_preserved=True,
        preserved_early_targets_checked=True,train_test_source_hash_overlap=0,numeric_house_overlap=0,
        video_observation_counts_checked=True,required_state_observation_counts_checked=True,
        raw_trajectories_alignment_checked=len(records),manifest_sha256=manifests,**counts))
    write(run/'status.json',dict(state='complete',**counts,training_started=False,model_evaluation_started=False,
        physical_grasp_retention_certified=False,official_val_used_as_test=True,separate_development_split=False))
    report(args)


def report(args):
    run=args.run;value=json.loads((run/'status.json').read_text())
    text=['# Official Train / official Val-as-Test','',
          'Two project splits only. Existing expert demonstrations are reused; this preparation does not generate experts, train a model or launch policy evaluation.',
          'Prior inspection is recorded per Test case. Source pose labels do not certify stable-grasp opening.','',
          '```json',json.dumps(value,indent=2),'```','',
          'Test cases are not selected by baseline outcomes. All previous source trajectories and pose references are preserved.',
          'Completing the public catalog may add no expert cases if a released package contains videos but no trajectory HDF5.',
          'Source scene compatibility and expert grasp-retention validation remain pending.']
    (run/'report.md').write_text('\n'.join(text)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Train / Test data</title><style>body{font:16px system-ui;margin:40px}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(text))+'</pre>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','collect','report'))
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    if args.command=='prepare' and args.config is None:parser.error('prepare requires --config')
    try:
        globals()[args.command](args)
    except Exception:
        if args.command=='collect':
            problem=dict(state='failed',error=traceback.format_exc(),training_started=False,
                model_evaluation_started=False,physical_grasp_retention_certified=False,
                all_integrity_checks_passed=False)
            write(args.run/'audit/integrity.json',problem)
            write(args.run/'status.json',problem)
            report(args)
        raise


if __name__=='__main__':main()
