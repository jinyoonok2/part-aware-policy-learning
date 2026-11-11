"""Export explicit initial-state EpisodeSpecs from released expert metadata.

Only frozen initial configuration is exported. Expert actions and target poses
are never placed in the policy input. Scene family must be matched by identities
in an actual scene XML; missing scenes remain pending instead of being guessed.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp.runtime import read_config
from data_preparation.prepare_test_set import digest, write

FAMILIES = ('procthor-objaverse', 'procthor-10k', 'holodeck-objaverse')
STORE = Path('/bigtemp/xna8aw/part-aware-policy-learning')


def episode_schema():
    """Load the native schema without importing its simulation runner package."""
    name = '_part_aware_native_episode_schema'
    if name not in sys.modules:
        package = importlib.util.find_spec('molmo_spaces')
        if package is None or not package.submodule_search_locations:
            raise ImportError('Install the pinned MolmoSpaces framework first')
        path = Path(next(iter(package.submodule_search_locations))) / 'evaluation/benchmark_schema.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].EpisodeSpec


def scene_matches(xml, task):
    names = {node.get('name') for node in ET.parse(xml).getroot().iter('body')}
    missing = set(task.get('object_poses', {})) - names - set(task.get('added_objects', {}))
    return task['door_body_name'] in names and not missing


def resolve_scene(row, config, assets, manager=None):
    split = row['source_split']
    matches, checked = [], []
    for family in FAMILIES:
        source = family + '-' + split
        xml = Path(assets) / 'scenes' / source / f'{split}_{row["house"]}_ceiling.xml'
        if not xml.is_file() and manager is not None:
            packages = manager.index_lookup('scenes', source, str(row['house']))
            if len(packages) == 1:
                manager.install_packages('scenes', {source: packages})
        if not xml.is_file():
            checked.append(dict(family=family, state='scene_not_staged'))
            continue
        matched = bool(scene_matches(xml, config['task_config']))
        receipt = dict(family=family, xml=str(xml.resolve()), xml_sha256=digest(xml),
                       identity_match=matched)
        checked.append(receipt)
        if matched:
            matches.append(receipt)
    if len(matches) != 1:
        raise ValueError(f'Expected one matching scene family; found {len(matches)}. '
                         'Stage the scene packages or resolve asset compatibility first.')
    return matches[0], checked


def make_episode(row, config, scene, family, seed=42):
    """Use the frozen robot/task initial state, independent of the expert path."""
    cameras = []
    fields = ('name', 'reference_body_names', 'camera_offset', 'lookat_offset',
              'camera_quaternion', 'fov', 'record_depth')
    for camera in config['camera_config']['cameras']:
        cameras.append(dict(type='robot_mounted', **{key: deepcopy(camera[key])
                           for key in fields if key in camera}))
    task = deepcopy(config['task_config'])
    task.update(task_cls='molmo_spaces.tasks.opening_tasks.DoorOpeningTask',
                task_type='door_opening')
    modifications = {key: deepcopy(task.get(key, [] if key == 'removed_objects' else {}))
                     for key in ('added_objects', 'object_poses', 'removed_objects')}
    value = dict(house_index=row['house'], scene_dataset=family, data_split=row['source_split'],
        seed=seed, robot=dict(robot_name='rby1m', init_qpos=deepcopy(config['robot_config']['init_qpos'])),
        img_resolution=[1024, 576], cameras=cameras, scene_modifications=modifications,
        task=task, language=dict(task_description=scene.get('task_description', row.get('task_description')),
            referral_expressions=deepcopy(scene.get('referral_expressions', {}))))
    EpisodeSpec = episode_schema()
    # Source HDF5 is recorded in the external manifest only, never needed by
    # the EpisodeSpec or the policy during online inference.
    return EpisodeSpec.model_validate(value).model_dump(exclude_none=True)


def export(args):
    original = json.loads(args.input.read_text())
    rows = original['cases'] if isinstance(original, dict) else original
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty source trajectory case manifest')
    source_case_count=len(rows)
    if args.limit:
        rows = rows[:args.limit]
    identities=[row.get('case_id',f'source_{index:04d}') for index,row in enumerate(rows)]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate source case IDs')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; source data are never overwritten')
    args.output.mkdir(parents=True)
    manager = None
    if args.stage_scenes:
        from molmospaces_resources import ResourceManager, R2RemoteStorage
        versions = json.loads((args.assets/'mjthor_data_type_to_source_to_versions.json').read_text())
        manager = ResourceManager(remote_storage=R2RemoteStorage('mujoco-thor-resources'),
            data_type_to_source_to_version=versions, cache_dir=args.cache, symlink_dir=args.assets)
    exported, pending = [], []
    for index, row in enumerate(rows):
        identity = row.get('case_id', f'source_{index:04d}')
        if not isinstance(identity, str) or Path(identity).name != identity or identity in ('', '.', '..'):
            raise ValueError('Case IDs must be safe directory names')
        try:
            dataset = Path(row.get('dataset_root', args.dataset))
            config, scene = read_config(dataset, row)
            match, checked = resolve_scene(row, config, args.assets, manager)
            episode = make_episode(row, config, scene, match['family'], args.seed)
            folder = args.output / 'cases' / identity
            write(folder / 'episode.json', episode)
            write(folder / 'scene_identity.json', dict(selected=match, candidates=checked))
            exported.append(dict(row, case_id=identity,
                episode_path=str((folder/'episode.json').relative_to(args.output)),
                episode_sha256=digest(folder/'episode.json'), scene_identity=match,
                stable_grasp_feasibility_verified=False, final_eligible=False,
                model_evaluation_ready=False, expert_runtime_inputs_used=False,
                source_qualification_status='pending_retained_grasp_verification'))
        except Exception as error:
            pending.append(dict(case_id=identity, house=row.get('house'), reason=str(error),
                                classification='source_or_scene_compatibility_pending'))
    status = dict(state='export_complete' if not pending else 'export_incomplete',
        source_manifest=str(args.input.resolve()), source_manifest_sha256=digest(args.input),
        source_case_count=source_case_count,diagnostic_subset=len(rows)!=source_case_count,
        requested_cases=len(rows), exported_cases=len(exported), pending_cases=len(pending),
        source_files_changed=False, physical_grasp_retention_certified=False,
        policy_input_contains_expert_actions_or_target_poses=False)
    write(args.output / 'status.json', status)
    write(args.output / 'pending_cases.json', dict(cases=pending))
    if exported and (not pending or args.allow_partial):
        write(args.output / 'policy_episode_manifest.json', dict(**status, cases=exported,
            partial_export=bool(pending), final_model_test_ready=False))
    print(json.dumps(status, indent=2))
    if pending and not args.allow_partial:
        raise RuntimeError('Some scenes are unresolved. No runnable policy manifest was emitted. '
                           'See pending_cases.json; --allow-partial is only for explicit diagnostics.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STORE/'datasets/interaction_pose_train_test_v1_20261107/final_episode_manifest.json')
    parser.add_argument('--dataset', type=Path, default=STORE/'datasets/interaction_pose_train_test_v1_20261107')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=STORE/'cache/molmospaces/assets')
    parser.add_argument('--cache', type=Path, default=STORE/'cache/molmo-spaces-resources')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Explicit diagnostic subset only')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--stage-scenes', action='store_true', help='Permit CPU scene-package downloads; no policy or physics execution')
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    export(args)


if __name__=='__main__':
    main()
