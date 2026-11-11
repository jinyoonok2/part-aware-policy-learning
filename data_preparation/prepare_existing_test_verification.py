"""Prepare and stage every released Test demonstration for measured replay.

The 197-case source inventory is preserved, including pose-ineligible cases.
Only initial configuration is exported into policy EpisodeSpecs. Original
commands remain separate, read-only expert-replay inputs.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.export_policy_episodes import make_episode, resolve_scene
from data_preparation.prepare_test_set import canonical, digest, write
from data_preparation.snapshots import freeze_packages
from grasp.metrics import FINAL_POLICY
from grasp.runtime import read_config

PROJECT = Path(__file__).resolve().parents[1]
STORE = Path('/bigtemp/xna8aw/part-aware-policy-learning')


def source_cases(manifest):
    document = json.loads(Path(manifest).read_text())
    cases = document['cases'] if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases:
        raise ValueError('A complete nonempty source Test inventory is required')
    ids = []
    for row in cases:
        identity = row.get('case_id')
        if not isinstance(identity, str) or Path(identity).name != identity or identity in ('', '.', '..'):
            raise ValueError('Case IDs must be safe directory names')
        if row.get('source_split') != 'val' or row.get('project_role', 'test') != 'test':
            raise ValueError('Existing Test verification requires official Val sources')
        ids.append(identity)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate existing Test case IDs')
    return cases


def prepare(args):
    rows = source_cases(args.manifest)
    if args.run.exists():
        raise FileExistsError('Use a fresh verification run; original data stay read-only')
    args.run.mkdir(parents=True)
    for folder in ('cases', 'source', 'logs', 'audit'):
        (args.run / folder).mkdir()
    cases = []
    for index, original in enumerate(rows):
        row = dict(original, source_ordinal=index, original_episode_index=None,
                   episode_path=f"cases/{original['case_id']}/episode.json")
        dataset = Path(row.get('dataset_root', args.dataset))
        config, scene = read_config(dataset, row)
        if float(scene.get('policy_dt_ms', 0)) != 100.:
            raise ValueError('Source policy period must be 100 ms; replay is not resampled')
        row['dataset_root'] = str(dataset.resolve())
        folder = args.run / 'cases' / row['case_id']
        folder.mkdir()
        write(folder / 'source_reference.json', row)
        write(folder / 'decoded_config.json', config)
        write(folder / 'decoded_scene.json', scene)
        row['source_config_sha256'] = digest(folder / 'decoded_config.json')
        row['source_scene_sha256'] = digest(folder / 'decoded_scene.json')
        write(folder / 'status.json', dict(state='queued_for_assets', case_id=row['case_id']))
        cases.append(row)
    hashes = freeze_packages(args.project, args.run / 'source')
    for name in ('prepare_existing_test_verification.py', 'existing_test_worker.py',
                 'report_benchmark_expert_verification.py'):
        target = args.run / 'source' / name
        shutil.copy2(args.project / 'data_preparation' / name, target)
        hashes[name] = digest(target)
    study = dict(name='All released official Val/Test expert-command verification',
        source_split='val', project_role='test', project_root=str(args.project.resolve()),
        source_manifest=str(args.manifest.resolve()), source_manifest_sha256=digest(args.manifest),
        existing_expert_test_trajectories=len(cases), cases=cases, source_hashes=hashes,
        metric_policy=dict(FINAL_POLICY), metric_version=FINAL_POLICY['metric_version'],
        metric_policy_sha256=canonical(FINAL_POLICY), horizon=400, maximum_executed_actions=400,
        continuation_steps=10, timing_ms=dict(policy=100., control=20., simulation=4.),
        audit_seed=42, replay_mode='source_commands_noise_disabled',
        original_source_rng_available=False, original_applied_control_ground_truth_available=False,
        original_exact_noise_replay_claim=False, stage_complete=False, worker_source_frozen=True,
        source_inventory_preserved=True, selection='Entire supplied official Val inventory, in existing order; no pose-eligibility or policy-outcome filtering',
        qualification_scope='New measured physical replay witness under source absolute controllers with noise disabled. A failed replay is not proof of task impossibility or source-trajectory invalidity.',
        training_started=False, model_evaluation_started=False)
    write(args.run / 'study.json', study)
    write(args.run / 'status.json', dict(state='prepared_pending_assets', expected_cases=len(cases),
        staged_cases=0, training_started=False, model_evaluation_started=False))
    print(json.dumps(dict(run=str(args.run), cases=len(cases), state='prepared_pending_assets'), indent=2))


def require_frozen(run, study):
    if digest(study['source_manifest']) != study['source_manifest_sha256']:
        raise ValueError('Original Test inventory changed')
    for relative, expected in study['source_hashes'].items():
        if digest(run / 'source' / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)


def stage(args):
    from molmospaces_resources import ResourceManager, R2RemoteStorage, split_query_tokens
    run = args.run
    study = json.loads((run / 'study.json').read_text())
    require_frozen(run, study)
    assets, cache = args.assets.resolve(), args.cache.resolve()
    tree = ast.parse((Path(study['project_root']) / 'molmospaces/molmo_spaces/molmo_spaces_constants.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'DATA_TYPE_TO_SOURCE_TO_VERSION' for target in node.targets))
    scope = {}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<pinned resource versions>', 'exec'), {'dict': dict}, scope)
    versions = scope['DATA_TYPE_TO_SOURCE_TO_VERSION']
    manager = ResourceManager(remote_storage=R2RemoteStorage('mujoco-thor-resources'),
                              data_type_to_source_to_version=versions, cache_dir=cache, symlink_dir=assets)
    original_cache = Path('/bigtemp/xna8aw/LRL/cache/molmo-spaces-resources')

    def install(dtype, name, packages):
        packages = sorted(set(packages))
        if not packages:
            return
        relative = Path(dtype) / name / versions[dtype][name]
        for package in packages:
            for leaf in manager.tries(dtype, name)[package].leaf_paths():
                destination, original = cache / relative / leaf, original_cache / relative / leaf
                if not destination.exists() and original.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + f'.source-staging-{os.getpid()}')
                    shutil.copy2(original, temporary)
                    temporary.replace(destination)
        manager.install_packages(dtype, {name: packages})

    # resolve_scene uses index_lookup/install_packages. Route the latter through
    # the copy-first adapter while preserving the native resource-manager API.
    class SceneManager:
        def index_lookup(self, *values):
            return manager.index_lookup(*values)

        def install_packages(self, dtype, sources):
            for name, packages in sources.items():
                install(dtype, name, packages)

    install('robots', 'rby1m', manager.find_all_packages_for_source('robots', 'rby1m'))
    staged, errors = [], []
    for row in study['cases']:
        folder = run / 'cases' / row['case_id']
        try:
            write(folder / 'status.json', dict(state='staging_source_scene_identity', case_id=row['case_id']))
            config = json.loads((folder / 'decoded_config.json').read_text())
            scene = json.loads((folder / 'decoded_scene.json').read_text())
            if digest(folder / 'decoded_config.json') != row['source_config_sha256'] or digest(folder / 'decoded_scene.json') != row['source_scene_sha256']:
                raise ValueError('Decoded source configuration changed')
            match, checked = resolve_scene(row, config, assets, SceneManager())
            # Resolve resource references in the logical assets tree rather
            # than the physical versioned cache targeted by its symlinks.
            xml = assets / 'scenes' / (match['family'] + '-val') / f"val_{row['house']}_ceiling.xml"
            objects, filenames = {}, set()
            resource_files = [node.get('file') for node in ET.parse(xml).getroot().iter() if node.get('file')]
            for filename in resource_files:
                relative = Path(os.path.normpath(xml.parent / filename)).relative_to(assets.resolve())
                if relative.parts[0] != 'objects':
                    continue
                name, leaf = relative.parts[1], Path(*relative.parts[2:])
                objects.setdefault(name, set()).update(manager.find_archives('objects', name, [leaf]))
                filenames.add(leaf.name)
            for filename in config['task_config'].get('added_objects', {}).values():
                relative = Path(filename)
                if relative.parts[0] != 'objects':
                    raise ValueError('Unsupported added-object resource path')
                name, leaf = relative.parts[1], Path(*relative.parts[2:])
                objects.setdefault(name, set()).update(manager.find_archives('objects', name, [leaf]))
                filenames.add(leaf.name)
            for name, packages in objects.items():
                install('objects', name, packages)
            grasps = set()
            for filename in filenames:
                for token in split_query_tokens(filename):
                    grasps.update(manager.index_lookup('grasps', 'droid_objaverse', token))
            install('grasps', 'droid_objaverse', grasps)
            episode = make_episode(row, config, scene, match['family'], study['audit_seed'])
            write(folder / 'episode.json', episode)
            row['episode_sha256'] = digest(folder / 'episode.json')
            row['scene_identity'] = match
            write(folder / 'scene_identity.json', dict(selected=match, candidates=checked))
            receipt = dict(state='passed', case_id=row['case_id'], episode_sha256=row['episode_sha256'],
                           source_config_sha256=row['source_config_sha256'],
                           source_h5_sha256=row['h5_sha256'], scene_identity=match)
            write(folder / 'staging.json', receipt)
            write(folder / 'status.json', dict(state='staged', case_id=row['case_id']))
            staged.append(row)
        except Exception:
            value = dict(case_id=row['case_id'], state='asset_or_source_compatibility_error',
                         error=traceback.format_exc(), feasibility_status='unknown')
            write(folder / 'status.json', value)
            errors.append(value)
        write(run / 'status.json', dict(state='staging_existing_sources', expected_cases=len(study['cases']),
            staged_cases=len(staged), asset_errors=len(errors), training_started=False, model_evaluation_started=False))
        print(f"Staged {len(staged)}/{len(study['cases'])}; errors={len(errors)}", flush=True)
    study['stage_complete'] = not errors and len(staged) == len(study['cases'])
    write(run / 'study.json', study)
    write(run / 'staging_status.json', dict(state='passed' if study['stage_complete'] else 'incomplete',
        expected_cases=len(study['cases']), staged_cases=len(staged), errors=errors))
    if study['stage_complete']:
        write(run / 'policy_episode_manifest.json', dict(cases=staged, source_manifest=study['source_manifest'],
            source_manifest_sha256=study['source_manifest_sha256'], source_case_count=len(staged),
            requested_cases=len(staged), pending_cases=0, diagnostic_subset=False, partial_export=False,
            final_model_test_ready=False, physical_grasp_retention_certified=False))
    write(run / 'status.json', dict(state='staged' if study['stage_complete'] else 'staging_incomplete',
        expected_cases=len(study['cases']), staged_cases=len(staged), asset_errors=len(errors),
        expert_execution_started=False, training_started=False, model_evaluation_started=False))
    if not study['stage_complete']:
        raise RuntimeError('Some source scenes are unresolved; no incomplete population was silently exported')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'stage'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=STORE / 'datasets/interaction_pose_train_test_v1_20261107/final_episode_manifest.json')
    parser.add_argument('--dataset', type=Path, default=STORE / 'datasets/interaction_pose_train_test_v1_20261107')
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--assets', type=Path, default=STORE / 'cache/molmospaces/assets')
    parser.add_argument('--cache', type=Path, default=STORE / 'cache/molmo-spaces-resources')
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
