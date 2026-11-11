"""Fault-injection checks on a prepared /tmp expansion, never source datasets."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import unittest

from data_preparation.expand_interaction_pose_dataset import audit


RUN = None


@contextmanager
def edited(path, content):
    original = path.read_bytes()
    try:
        path.write_text(content)
        yield
    finally:
        path.write_bytes(original)


class ExpansionGuards(unittest.TestCase):
    def setUp(self):
        if RUN is None:
            self.skipTest('Use --prepared-run with an audited disposable /tmp expansion for fault injection')

    def test_duplicate_train_trajectory_rejected(self):
        path = RUN / 'pose_targets.jsonl'
        first = path.read_text().splitlines()[0]
        with edited(path, path.read_text() + first + '\n'):
            with self.assertRaisesRegex(AssertionError, 'Duplicate train trajectory'):
                audit(RUN)

    def test_validation_house_in_train_rejected(self):
        path = RUN / 'pose_targets.jsonl'
        row = json.loads((RUN / 'val_pose_targets.jsonl').read_text().splitlines()[0])
        row['source_split'] = 'train'
        with edited(path, path.read_text() + json.dumps(row) + '\n'):
            with self.assertRaises(AssertionError):
                audit(RUN)

    def test_changed_validation_index_rejected(self):
        path = RUN / 'val_pose_targets.jsonl'
        with edited(path, ' ' + path.read_text()):
            with self.assertRaises(AssertionError):
                audit(RUN)

    def test_changed_frozen_source_rejected(self):
        path = RUN / 'source/expand_interaction_pose_dataset.py'
        with edited(path, path.read_text() + '\n# injected mutation\n'):
            with self.assertRaisesRegex(AssertionError, 'Frozen source changed'):
                audit(RUN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-run', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    RUN = args.prepared_run.resolve()
    assert RUN.is_relative_to(Path('/tmp')), 'Fault injection is restricted to a disposable /tmp preparation'
    assert (RUN / 'audit/expansion_integrity.json').exists(), 'Run the normal partial audit first'
    unittest.main(argv=['test_pose_data_expansion.py', *remaining])
