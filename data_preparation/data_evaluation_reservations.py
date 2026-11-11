"""Freeze final-evaluation house reservations when preparing new source data.

Preparation consults the project registry. Collection/resume consults only its
own immutable snapshot, never a subsequently edited workspace registry.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import hashlib
import json
from pathlib import Path
import re
import shutil


SNAPSHOT = 'final_evaluation_reservations.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def registry_house_ids(document):
    if document.get('schema_version') != 1 or not isinstance(document.get('evaluations'), list):
        raise ValueError('Invalid final-evaluation reservations schema')
    houses = set()
    for evaluation in document['evaluations']:
        values = evaluation.get('house_ids')
        if (not isinstance(values, list) or not all(type(h) is int and h >= 0 for h in values)
                or len(values) != len(set(values))):
            raise ValueError('Invalid reserved final-evaluation house IDs')
        expected = evaluation.get('episode_manifest_sha256')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Missing final-evaluation manifest identity')
        if not isinstance(evaluation.get('run'), str):
            raise ValueError('Missing final-evaluation run path')
        filename = evaluation.get('manifest_filename', 'final_episode_manifest.json')
        if not isinstance(filename, str) or Path(filename).name != filename or filename in ('', '.', '..'):
            raise ValueError('Reservation manifest must be a filename within its run')
        houses.update(values)
    return sorted(houses)


def load_reservations(project, registry=None):
    path = Path(registry) if registry is not None else Path(project) / 'configs' / SNAPSHOT
    if not path.exists():
        document = dict(schema_version=1, evaluations=[])
        return dict(registry_present=False, registry_source_path=str(path.resolve()),
                    registry_sha256=None, house_ids=[], document=document)
    document = json.loads(path.read_text())
    houses = registry_house_ids(document)
    for evaluation in document['evaluations']:
        manifest = Path(evaluation['run']) / evaluation.get('manifest_filename', 'final_episode_manifest.json')
        if sha256(manifest) != evaluation['episode_manifest_sha256']:
            raise ValueError('Final-evaluation manifest changed: ' + str(manifest))
        case_houses = sorted({row['house'] for row in json.loads(manifest.read_text())['cases']})
        if case_houses != sorted(evaluation['house_ids']):
            raise ValueError('Reserved house IDs do not match final-evaluation manifest')
    return dict(registry_present=True, registry_source_path=str(path.resolve()),
                registry_sha256=sha256(path), house_ids=houses, document=document)


def record_house(record):
    if type(record.get('house')) is int:
        return record['house']
    match = re.search(r'(?:^|/)house_(\d+)(?:/|$)', record.get('h5', ''))
    if match:
        return int(match[1])
    raise ValueError('Cannot establish source house for reserved-evaluation audit')


def require_source_disjoint(records, reservations, context):
    overlap = {record_house(row) for row in records} & set(reservations['house_ids'])
    if overlap:
        raise ValueError(context + ' overlaps reserved final-evaluation house IDs: ' + str(sorted(overlap)))


def exclude_reserved_order(order, reservations):
    reserved = set(reservations['house_ids'])
    return [row for row in order if row['house'] not in reserved]


def freeze_reservations(run, reservations):
    destination = Path(run) / 'source' / SNAPSHOT
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reservations['registry_present']:
        path = Path(reservations['registry_source_path'])
        if sha256(path) != reservations['registry_sha256']:
            raise ValueError('Reservation registry changed during data preparation')
        shutil.copy2(path, destination)
    else:
        destination.write_text(json.dumps(reservations['document'], indent=2) + '\n')
    result = {k:v for k,v in reservations.items() if k != 'document'}
    result.update(snapshot_filename=SNAPSHOT, snapshot_sha256=sha256(destination))
    return result


def check_frozen_reservations(run, study, order=None):
    reservation = study.get('final_evaluation_reservations')
    if reservation is None:
        return  # Legacy frozen studies retain their original contract.
    snapshot = Path(run) / 'source' / reservation['snapshot_filename']
    if sha256(snapshot) != reservation['snapshot_sha256']:
        raise ValueError('Frozen final-evaluation reservation snapshot changed')
    if registry_house_ids(json.loads(snapshot.read_text())) != reservation['house_ids']:
        raise ValueError('Frozen final-evaluation house list changed')
    if order is not None:
        require_source_disjoint(order, reservation, 'Frozen source collection order')
