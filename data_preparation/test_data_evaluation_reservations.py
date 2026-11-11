"""Reservation guards prevent future source collection from consuming final houses."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import data_preparation.collect_interaction_pose_dataset as collector
import data_preparation.expand_interaction_pose_dataset as expansion
from data_preparation.data_evaluation_reservations import (check_frozen_reservations, exclude_reserved_order,
    freeze_reservations, load_reservations, require_source_disjoint, sha256)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ReservationChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.eval=self.root/'evaluation'
        write(self.eval/'final_episode_manifest.json',dict(cases=[dict(house=2),dict(house=4)]))
        self.registry=self.root/'reservations.json'
        write(self.registry,dict(schema_version=1,evaluations=[dict(run=str(self.eval),
              episode_manifest_sha256=sha256(self.eval/'final_episode_manifest.json'),house_ids=[2,4])]))
        self.reservations=load_reservations(self.root,self.registry)

    def tearDown(self):self.temp.cleanup()

    def test_order_filter_preserves_rank_and_cross_family_reservations(self):
        order=[dict(house=i,rank=20+i,scene_dataset='other_family')for i in range(1,6)]
        kept=exclude_reserved_order(order,self.reservations)
        self.assertEqual([(r['house'],r['rank'])for r in kept],[(1,21),(3,23),(5,25)])

    def test_source_reuse_fails_for_train_and_val_and_h5_fallback(self):
        for rows,context in [([dict(house=2)],'Train'),([dict(house=4)],'Val'),
                             ([dict(h5='raw/part0/train/house_2/batch.h5')],'Train')]:
            with self.subTest(context=context,rows=rows):
                with self.assertRaisesRegex(ValueError,'reserved final-evaluation'):
                    require_source_disjoint(rows,self.reservations,context)

    def test_registry_house_mismatch_and_manifest_mutation_fail_closed(self):
        original=json.loads(self.registry.read_text())
        changed=json.loads(self.registry.read_text());changed['evaluations'][0]['house_ids']=[2]
        write(self.registry,changed)
        with self.assertRaisesRegex(ValueError,'do not match'):load_reservations(self.root,self.registry)
        write(self.registry,original);write(self.eval/'final_episode_manifest.json',dict(cases=[dict(house=5)]))
        with self.assertRaisesRegex(ValueError,'manifest changed'):load_reservations(self.root,self.registry)

    def test_resume_uses_snapshot_even_if_live_registry_changes(self):
        run=self.root/'run';study=dict(final_evaluation_reservations=freeze_reservations(run,self.reservations))
        write(self.registry,dict(invalid_live_document=True))
        check_frozen_reservations(run,study,[dict(house=1)])
        with self.assertRaisesRegex(ValueError,'overlaps reserved'):
            check_frozen_reservations(run,study,[dict(house=2)])
        (run/'source/final_evaluation_reservations.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'snapshot changed'):check_frozen_reservations(run,study)

    def source_fixture(self):
        run=self.root/'data';(run/'metadata').mkdir(parents=True)
        for split in ('train','val'):(run/f'metadata/{split}_pkgs.parquet').write_bytes(split.encode())
        val=self.root/'val'
        write(val/'valid_trajectory_index.json',{'house_900':{'house_900/batch.h5':{f'traj_{i}':10 for i in range(197)}}})
        config=self.root/'config.json'
        write(config,dict(source_split='train',seed=9,validation_root=str(val),reservation_registry=str(self.registry)))
        packages=[dict(path=f'DoorOpeningDataGenConfig_house_{h}.tar.zst',inflated_size=1,part=0)for h in range(1,6)]
        return run,config,packages

    def test_fresh_collector_filters_final_houses_and_resumes_its_frozen_list(self):
        run,config,packages=self.source_fixture()
        with patch.object(collector.pq,'read_table',return_value=SimpleNamespace(to_pylist=lambda:packages)):
            collector.prepare(run,config)
        order=json.loads((run/'house_order.json').read_text())
        self.assertEqual({r['house']for r in order},{1,3,5})
        study=json.loads((run/'study.json').read_text())
        self.assertEqual(study['final_evaluation_reservations']['house_ids'],[2,4])
        self.assertTrue((run/'source/data_evaluation_reservations.py').is_file())
        write(self.registry,dict(invalid_live_document=True))
        with patch.object(collector.pq,'read_table',side_effect=AssertionError('metadata selection must not rerun')):
            collector.prepare(run,config)

    def test_fresh_collector_allows_official_val_as_reserved_test(self):
        run,config,packages=self.source_fixture()
        cfg=json.loads(config.read_text());val=Path(cfg['validation_root'])
        write(val/'valid_trajectory_index.json',{'house_2':{'house_2/batch.h5':{f'traj_{i}':10 for i in range(197)}}})
        with patch.object(collector.pq,'read_table',return_value=SimpleNamespace(to_pylist=lambda:packages)):
            collector.prepare(run,config)
        self.assertTrue((run/'study.json').exists())
        self.assertNotIn(2,{r['house'] for r in json.loads((run/'house_order.json').read_text())})

    def expansion_fixture(self,train_house=1,val_house=900):
        parent=self.root/'parent';parent.mkdir()
        val=self.root/'val';write(val/'valid_trajectory_index.json',{})
        for split in ('train','val'):
            (parent/'metadata').mkdir(exist_ok=True)
            (parent/f'metadata/{split}_pkgs.parquet').write_bytes(split.encode())
        previous=dict(revision='abc',validation_root=str(val),validation_index_sha256=sha256(val/'valid_trajectory_index.json'),
                      metadata_sha256={s:sha256(parent/f'metadata/{s}_pkgs.parquet')for s in ('train','val')})
        write(parent/'study.json',previous)
        write(parent/'status.json',dict(state='complete',pose_trajectories=503,houses=1,download_bytes=1))
        write(parent/'audit/dataset_integrity.json',dict(state='passed'))
        order=[dict(rank=i,house=h,packages=[dict(size=1,inflated_size=1)])for i,h in enumerate((train_house,2,3,4))]
        write(parent/'house_order.json',order)
        for split,house in [('train',train_house),('val',val_house)]:
            for filename in [f'{split}_trajectories.jsonl', 'pose_targets.jsonl' if split=='train' else 'val_pose_targets.jsonl']:
                (parent/filename).write_text(json.dumps(dict(house=house))+'\n')
        write(parent/'validation_status.json',{})
        source=parent/'source/collect_interaction_pose_dataset.py';source.parent.mkdir();source.write_text('# inert extractor fixture\n')
        write(parent/'source-sha256.json',{'collect_interaction_pose_dataset.py':sha256(source)})
        (parent/'raw').mkdir()
        config=self.root/'expand.json'
        write(config,dict(parent_dataset=str(parent),source_revision='abc',target_pose_trajectories=1000,
              exclude_validation_numeric_house_ids=True,maximum_houses=4,download_budget_bytes=1000,
              advertised_inflated_budget_bytes=1000,name='fixture',label_status='offline',scope='test',
              checkpoint_report_interval_houses=5,reservation_registry=str(self.registry)))
        return argparse.Namespace(config=config,project=Path(collector.__file__).resolve().parents[1],run=self.root/'expanded')

    def test_expansion_filters_new_order_and_freezes_reservation_provenance(self):
        args=self.expansion_fixture()
        with patch.object(expansion,'check_parent'),patch.object(expansion,'parent_identity',return_value={}),patch.object(expansion,'publish'):
            expansion.prepare(args)
        self.assertEqual([r['house']for r in json.loads((args.run/'selection_order.json').read_text())],[1,3])
        study=json.loads((args.run/'study.json').read_text())
        self.assertEqual(study['final_evaluation_reservations']['house_ids'],[2,4])
        self.assertIn('excluded_final_evaluation_house_ids.json',study['selection_file_sha256'])
        self.assertTrue((args.run/'source/data_evaluation_reservations.py').is_file())
        check_frozen_reservations(args.run,study,json.loads((args.run/'selection_order.json').read_text()))

    def test_expansion_rejects_parent_train_overlap_before_creating_output(self):
        args=self.expansion_fixture(train_house=2)
        with self.assertRaisesRegex(ValueError,'Reused parent Train inventory overlaps reserved'):
            expansion.prepare(args)
        self.assertFalse(args.run.exists())

    def test_expansion_allows_official_val_as_reserved_test(self):
        args=self.expansion_fixture(val_house=4)
        with patch.object(expansion,'check_parent'),patch.object(expansion,'parent_identity',return_value={}),patch.object(expansion,'publish'):
            expansion.prepare(args)
        self.assertTrue(args.run.exists())
        self.assertNotIn(4,{r['house'] for r in json.loads((args.run/'selection_order.json').read_text())})

    def test_explicit_test_reservation_manifest_and_unsafe_path(self):
        manifest=self.eval/'test_house_reservation_manifest.json'
        write(manifest,dict(cases=[dict(house=3)]))
        document=dict(schema_version=1,evaluations=[dict(run=str(self.eval),manifest_filename=manifest.name,
              episode_manifest_sha256=sha256(manifest),house_ids=[3])])
        write(self.registry,document)
        self.assertEqual(load_reservations(self.root,self.registry)['house_ids'],[3])
        document['evaluations'][0]['manifest_filename']='../outside.json'
        write(self.registry,document)
        with self.assertRaisesRegex(ValueError,'within its run'):
            load_reservations(self.root,self.registry)

    def test_expansion_rejects_reserved_orphan_raw_house_even_if_not_indexed(self):
        args=self.expansion_fixture()
        (self.root/'parent/raw/part0/train/house_2').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Reused parent raw house directories overlaps reserved'):
            expansion.prepare(args)
        self.assertFalse(args.run.exists())


if __name__=='__main__':unittest.main()
