"""Guard independent-cohort membership and frozen imports without simulation."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from data_preparation.prepare_test_set import plan, digest
from data_preparation.test_bounded_benchmark_verification import fixture


class TestSetPlanChecks(unittest.TestCase):
    def test_standalone_keeps_an_independent_fixed_population(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);train=root/'train';existing=root/'existing'
            train.mkdir();existing.mkdir()
            (train/'train_trajectories.jsonl').write_text(json.dumps(dict(house=10000))+'\n')
            (existing/'final_episode_manifest.json').write_text(json.dumps(dict(cases=[dict(house=2)])))
            _,episodes=fixture(houses=6)
            benchmark=root/'benchmark.json';benchmark.write_text(json.dumps(episodes))
            args=Namespace(run=root/'fixed',train_dataset=train,existing_dataset=existing,
                total=8,benchmark=benchmark,seed=6198,candidate_houses=4,batch_size=4,
                project=Path(__file__).resolve().parents[1],curobo_receipt=root/'receipt.json',standalone=True)
            plan(args)
            study=json.loads((args.run/'study.json').read_text())
            self.assertEqual(study['target_source_episodes'],8)
            self.assertEqual(study['existing_cases_included'],0)
            self.assertIsNone(study['target_qualified_test_episodes'])
            self.assertNotIn(2,{row['house'] for row in study['cases']})
            self.assertFalse(study['replace_failed_experts'])
            self.assertTrue((args.run/'source/pose_learning/data.py').exists())
            self.assertEqual(len(json.loads((args.run/'policy_episode_manifest.json').read_text())['cases']),8)

    def test_default_keeps_existing_cohort_separate_and_freezes_package_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, existing = root/'train', root/'existing'
            train.mkdir(); existing.mkdir()
            (train/'train_trajectories.jsonl').write_text(json.dumps(dict(house=10000))+'\n')
            existing_path = existing/'final_episode_manifest.json'
            existing_path.write_text(json.dumps(dict(cases=[dict(house=20000+i) for i in range(197)])))
            prior = digest(existing_path)
            _, episodes = fixture(houses=6)
            benchmark = root/'benchmark.json';benchmark.write_text(json.dumps(episodes))
            args = Namespace(run=root/'plan', train_dataset=train, existing_dataset=existing,
                total=8, benchmark=benchmark, seed=6198, candidate_houses=4, batch_size=4,
                project=Path(__file__).resolve().parents[1], curobo_receipt=root/'receipt.json')
            plan(args)
            study=json.loads((args.run/'study.json').read_text())
            self.assertEqual(study['existing_expert_test_trajectories'],197)
            self.assertIsNone(study['additional_needed_if_all_existing_qualify'])
            self.assertIsNone(study['target_qualified_test_episodes'])
            self.assertEqual(study['existing_cases_included'],0)
            self.assertEqual(len(study['cases']),8)
            self.assertFalse(study['existing_strict_qualification_verified'])
            self.assertEqual(digest(existing_path),prior)
            self.assertTrue((args.run/'source/grasp/runtime.py').is_file())
            self.assertTrue((args.run/'source/data_preparation/prepare_test_set.py').is_file())
            subprocess.run([sys.executable,str(args.run/'source/verify_benchmark_expert.py'),'--help'],
                cwd=root,check=True,capture_output=True)
            for relative, expected in study['source_hashes'].items():
                self.assertEqual(digest(args.run/'source'/relative),expected)
            with self.assertRaises(FileExistsError):
                plan(args)


if __name__=='__main__':
    unittest.main()
