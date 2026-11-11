"""Stage an isolated one-scene smoke run from existing, read-only LRL resources."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--house-indices', type=int, nargs='+', default=[0])
    parser.add_argument('--episodes-per-house', type=int, default=1)
    parser.add_argument('--episode-indices', type=int, nargs='+')
    parser.add_argument('--project-root', type=Path)
    args = parser.parse_args()
    root = args.project_root or Path(__file__).resolve().parents[1]
    store = Path('/bigtemp/xna8aw/part-aware-policy-learning')
    source = Path('/bigtemp/xna8aw/LRL/cache/molmo-spaces-resources')
    cache = store/'cache/molmo-spaces-resources'
    assets = store/'cache/molmospaces/assets'
    args.output.mkdir(parents=True, exist_ok=False)
    tree = ast.parse((root/'molmospaces/molmo_spaces/molmo_spaces_constants.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DATA_TYPE_TO_SOURCE_TO_VERSION' for t in n.targets))
    scope = {}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<source versions>', 'exec'), {'dict': dict}, scope)
    versions = scope['DATA_TYPE_TO_SOURCE_TO_VERSION']
    from molmospaces_resources import ResourceManager, R2RemoteStorage
    manager = ResourceManager(remote_storage=R2RemoteStorage('mujoco-thor-resources'), data_type_to_source_to_version=versions, cache_dir=cache, symlink_dir=assets)
    copied = []

    def copy(src, dst):
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        temporary = dst.with_name(dst.name+'.staging')
        shutil.copy2(src, temporary)
        temporary.replace(dst)
        copied.append({'source': str(src), 'destination': str(dst), 'bytes': dst.stat().st_size})

    # Source indices describe available packages, not a claim that all are extracted.
    for dtype, sources in versions.items():
        for name, version in sources.items():
            rel = Path(dtype)/name/version
            for meta in ['mjthor_resource_file_to_size_mb.json', 'mjthor_resources_combined_meta.json.gz']:
                copy(source/rel/meta, cache/rel/meta)
            install = assets/dtype/name
            if install.is_symlink():
                assert install.resolve().is_relative_to(cache.resolve()), install
            elif not install.exists():
                manager._create_install_links(dtype, name, manager.behavior(dtype, name), cache/rel, install)
    (assets/'mjthor_data_type_to_source_to_versions.json').write_text(json.dumps(versions, indent=2))
    (cache/'mjthor_data_type_to_source_to_versions.json').write_text(json.dumps({d: {s: [v] for s, v in ss.items()} for d, ss in versions.items()}, indent=2))

    def stage_packages(dtype, name, packages):
        rel = Path(dtype)/name/versions[dtype][name]
        for package in sorted(set(packages)):
            trie = manager.tries(dtype, name)[package]
            for path in trie.leaf_paths():
                if (source/rel/path).exists():
                    copy(source/rel/path, cache/rel/path)
                elif not (cache/rel/path).exists():
                    print(f'Native resource manager will fetch missing package: {dtype}/{name}/{package}', flush=True)
                    break
        manager.install_packages(dtype, {name: sorted(set(packages))})

    stage_packages('robots', 'rby1m', manager.find_all_packages_for_source('robots', 'rby1m'))
    object_packages = set()
    object_filenames = set()
    # Follow only XML file references needed by the selected scene.
    for house in args.house_indices:
        scene_packages = manager.index_lookup('scenes', 'procthor-10k-val', str(house))
        assert len(scene_packages) == 1, scene_packages
        stage_packages('scenes', 'procthor-10k-val', scene_packages)
        scene = assets/f'scenes/procthor-10k-val/val_{house}_ceiling.xml'
        for node in ET.parse(scene).getroot().iter():
            filename = node.get('file')
            if not filename:
                continue
            target = scene.parent/filename
            asset_relative = Path(os.path.normpath(target)).relative_to(assets)
            if asset_relative.parts[0] != 'objects':
                continue
            name = asset_relative.parts[1]
            leaf = Path(*asset_relative.parts[2:])
            if name == 'thor':
                copy(source/'objects/thor'/versions['objects']['thor']/leaf, cache/'objects/thor'/versions['objects']['thor']/leaf)
            else:
                object_filenames.add(leaf.name)
                object_packages.update((name, pkg) for pkg in manager.find_archives('objects', name, [leaf]))
    for name in sorted({n for n, _ in object_packages}):
        stage_packages('objects', name, [p for n, p in object_packages if n == name])
    # The sampler requests grasp files even for a learned-policy rollout.
    grasp_packages = set()
    from molmospaces_resources import split_query_tokens
    for filename in object_filenames:
        for token in split_query_tokens(filename):
            grasp_packages.update(manager.index_lookup('grasps', 'droid_objaverse', token))
    stage_packages('grasps', 'droid_objaverse', grasp_packages)
    # Copy the existing metadata database, never share its writable lock file.
    lmdb_source = Path('/bigtemp/xna8aw/LRL/cache/molmospaces/assets/.lmdb/objathor_metadata/20260129/data.mdb')
    copy(lmdb_source, assets/'.lmdb/objathor_metadata/20260129/data.mdb')
    benchmark = source/'mujoco/benchmarks/molmospaces-bench-v2/20260327/procthor-10k/rby1_benchmarks/door_opening_benchmark/benchmark.json'
    episodes = json.loads(benchmark.read_text())
    selected = []
    if args.episode_indices is not None:
        assert len(args.episode_indices) == len(set(args.episode_indices))
        selected = [(i, episodes[i]) for i in args.episode_indices]
        assert {e['house_index'] for _, e in selected} == set(args.house_indices)
    else:
        for house in args.house_indices:
            matching = [(i, e) for i, e in enumerate(episodes) if e['house_index'] == house]
            assert len(matching) >= args.episodes_per_house, (house, len(matching))
            selected.extend(matching[:args.episodes_per_house])
    (args.output/'benchmark').mkdir()
    (args.output/'benchmark/benchmark.json').write_text(json.dumps([e for _, e in selected], indent=2))
    checkpoint = Path('/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask')
    manifest = {'stage': 'prepared', 'benchmark_source': str(benchmark), 'benchmark_sha256': hashlib.sha256(benchmark.read_bytes()).hexdigest(), 'original_episode_index': 0, 'episode': episodes[0], 'checkpoint': str(checkpoint), 'checkpoint_config_sha256': hashlib.sha256((checkpoint/'config.yaml').read_bytes()).hexdigest(), 'copied_files': copied, 'copied_bytes': sum(x['bytes'] for x in copied), 'source_roots_read_only_by_convention': [str(source), str(lmdb_source.parent)], 'asset_scope': 'source metadata plus scene 0, its robot/object/grasp dependencies and metadata DB; other scenes are not staged'}
    manifest.update(original_episode_index=selected[0][0], episode=selected[0][1], original_episode_indices=[i for i, _ in selected], house_indices=args.house_indices, selection_rule='First N benchmark entries per listed house, fixed before evaluating policy outcomes', asset_scope='Source metadata, listed houses and required robot/object/grasp dependencies; other scenes are not staged')
    if args.episode_indices is not None:
        manifest['selection_rule'] = 'Exact episode indices from the frozen cohort manifest'
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(args.output), 'copied_files': len(copied), 'copied_bytes': manifest['copied_bytes']}), flush=True)


if __name__ == '__main__':
    main()
