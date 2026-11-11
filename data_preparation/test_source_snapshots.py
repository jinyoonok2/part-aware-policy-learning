"""Protect legacy-source readability and the official-Val Test role."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py

from data_preparation.collect_interaction_pose_dataset import digest, index_validation
from data_preparation.snapshots import load_source_module


class SourceSnapshotChecks(unittest.TestCase):
    def test_historical_flat_import_is_supported_without_workspace_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'extractor.py'
            path.write_text('from data_evaluation_reservations import require_source_disjoint\nVALUE = 7\n')
            prior=sys.modules.get('data_evaluation_reservations')
            module=load_source_module('test_historical_extractor',path)
            self.assertEqual(module.VALUE,7)
            self.assertIs(sys.modules.get('data_evaluation_reservations'),prior)
            module.require_source_disjoint([dict(house=1)],dict(house_ids=[2]),'Train')

    def test_fresh_collection_preserves_all_test_sources_and_marks_label_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'released';source.mkdir()
            (source/'house_3').mkdir()
            h5=source/'house_3/trajectory.h5'
            with h5py.File(h5,'w') as data:
                data.create_group('traj_0');data.create_group('traj_1')
            inventory=source/'valid_trajectory_index.json'
            inventory.write_text(json.dumps({'house_3':{'house_3/trajectory.h5':{'traj_0':2,'traj_1':2}}}))
            study=dict(validation_root=str(source),validation_index_sha256=digest(inventory),
                       validation_trajectory_count=2)
            run=root/'prepared';run.mkdir();raw_hash=digest(h5)
            def inspect(run,study,house,part,path,group,valid):
                row=dict(source_split='train',house=house,h5=str(path.relative_to(run)),
                         trajectory=group.name.strip('/'))
                return row,dict(row) if group.name=='/traj_0' else None,'no eligible pose' if group.name=='/traj_1' else None
            with patch('data_preparation.collect_interaction_pose_dataset.inspect_trajectory',side_effect=inspect):
                index_validation(run,study)
            records=[json.loads(line)for line in(run/'val_trajectories.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),2)
            self.assertTrue(all(row['source_split']=='val' for row in records))
            self.assertEqual(sum(row['pose_eligible']for row in records),1)
            self.assertEqual(digest(h5),raw_hash)
            self.assertFalse(json.loads((run/'validation_status.json').read_text())['physical_grasp_retention_certified'])


if __name__=='__main__':
    unittest.main()
