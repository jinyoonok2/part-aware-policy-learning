"""Freeze every staged policy start without selecting on model outcomes."""
import argparse
import json
from pathlib import Path
import shutil

from .data import digest, write_json


def freeze_manifest(source, expected_source, output):
    document = json.loads(Path(source).read_text())
    expected = json.loads(Path(expected_source).read_text())
    rows = document['cases']
    originals = expected['cases'] if isinstance(expected, dict) else expected
    identities = [row['case_id'] for row in originals]
    if (document.get('partial_export') or document.get('diagnostic_subset')
            or document.get('pending_cases', 0)
            or [row['case_id'] for row in rows] != identities
            or len(set(identities)) != len(identities)):
        raise ValueError('Policy export must preserve the complete source population and order')
    checked = []
    for row in rows:
        identity = row['case_id']
        if Path(identity).name != identity or identity in ('', '.', '..'):
            raise ValueError('Unsafe case ID')
        path = Path(row['episode_path'])
        if not path.is_absolute():
            path = Path(source).parent / path
        if digest(path) != row['episode_sha256']:
            raise ValueError('Episode changed: ' + identity)
        checked.append((row, path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = []
    for row, path in checked:
        target = output / 'cases' / row['case_id'] / 'episode.json'
        target.parent.mkdir(parents=True)
        shutil.copy2(path, target)
        frozen.append(dict(row, episode_path=str(target.relative_to(output))))
    result = dict(cases=frozen, source_manifest=str(Path(source).resolve()),
        source_manifest_sha256=digest(source), source_inventory=str(Path(expected_source).resolve()),
        source_inventory_sha256=digest(expected_source), source_case_count=len(frozen),
        requested_cases=len(frozen), pending_cases=0, diagnostic_subset=False,
        partial_export=False, physical_grasp_retention_certified=False,
        policy_input_contains_expert_actions_or_target_poses=False,
        qualification_scope='All released source starts; strict expert feasibility remains unverified')
    write_json(output / 'policy_episode_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze_manifest(args.input, args.expected_source, args.output)
    print(json.dumps(dict(state='complete', cases=len(result['cases']), output=str(args.output))))


if __name__ == '__main__':
    main()
