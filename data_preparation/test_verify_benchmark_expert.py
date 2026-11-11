"""Identity, live-state and budget failure-injection checks without a GPU."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_preparation.verify_benchmark_expert import checked_existing,configuration_for_json,continuation_count,digest,initial_alignment,request_expert_action,verify_assets,verify_inputs,write


class ExactBenchmarkExpertChecks(unittest.TestCase):
    def test_native_planner_exhaustion_ends_attempt_but_unexpected_errors_propagate(self):
        class Expert:
            def __init__(self, error=None): self.error=error; self.observations=[]
            def get_action(self, observation):
                self.observations.append(observation)
                if self.error is not None: raise self.error
                return dict(left_gripper=[-.02])
        expert=Expert(ValueError('Max planning reattempts reached.'))
        action,terminal=request_expert_action(expert,{'frame':17})
        self.assertIsNone(action)
        self.assertEqual(terminal['reason'],'planner_retry_limit')
        self.assertEqual(expert.observations,[{'frame':17}])
        for error in (ValueError('Invalid planner state'),RuntimeError('Max planning reattempts reached.')):
            with self.assertRaises(type(error)):
                request_expert_action(Expert(error),{'frame':17})
        action,terminal=request_expert_action(Expert(),{'frame':0})
        self.assertEqual(action,dict(left_gripper=[-.02]))
        self.assertIsNone(terminal)

    def test_unbounded_config_snapshot_preserves_live_settings_and_finite_metrics(self):
        config=dict(limit=float('inf'), bounds=np.array([-np.inf, 2.5]), nested=[dict(enabled=True)])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'config.json'
            write(path, configuration_for_json(config))
            saved=json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertEqual(saved, dict(limit='Infinity', bounds=['-Infinity', 2.5], nested=[dict(enabled=True)]))
            self.assertTrue(np.isposinf(config['limit']))
            self.assertTrue(np.isneginf(config['bounds'][0]))
            with self.assertRaises(ValueError): write(path, dict(measured_error=float('inf')))
            with self.assertRaisesRegex(ValueError, 'NaN'):
                configuration_for_json(dict(limit=float('nan')))

    def fixture(self):
        return dict(robot=dict(init_qpos=dict(left_arm=[0.,1.])),
                    task=dict(robot_base_pose=[1.,2.,0.,1.,0.,0.,0.],articulated_joint_reset_state=[0.]),
                    scene_modifications=dict(object_poses=dict(handle=[1.,3.,1.,1.,0.,0.,0.])))

    def live(self):
        base=np.eye(4);base[:3,3]=[1.,2.,0.]
        handle=np.eye(4);handle[:3,3]=[1.,3.,1.]
        return dict(left_arm=np.array([0.,1.])),base,0.,dict(handle=handle)

    def test_live_base_perturbation_rejected_even_when_source_spec_unchanged(self):
        values=self.live();self.assertTrue(initial_alignment(self.fixture(),*values)['passed'])
        values[1][0,3]+=.1
        self.assertFalse(initial_alignment(self.fixture(),*values)['passed'])

    def test_robot_and_door_initial_mismatch_rejected(self):
        values=self.live();values[0]['left_arm'][1]+=.02
        self.assertFalse(initial_alignment(self.fixture(),*values)['passed'])
        values=self.live()
        self.assertFalse(initial_alignment(self.fixture(),values[0],values[1],.1,values[3])['passed'])

    def test_continuation_never_exceeds_shared_budget(self):
        self.assertEqual(continuation_count(395,400,10),5)
        self.assertEqual(continuation_count(400,400,10),0)
        with self.assertRaises(ValueError):continuation_count(401,400,10)

    def test_completed_witness_corruption_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'trace.json').write_text('[]')
            (root/'summary.json').write_text(json.dumps(dict(state='complete',episode_sha256='abc',artifact_sha256={'trace.json':digest(root/'trace.json')})))
            self.assertEqual(checked_existing(root,'abc')['state'],'complete')
            (root/'trace.json').write_text('[1]')
            with self.assertRaisesRegex(ValueError,'artifact changed'):checked_existing(root,'abc')

    def test_changed_episode_and_train_overlap_fail_before_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'source').mkdir();(root/'episode.json').write_text('{}')
            train=root/'train.jsonl';train.write_text(json.dumps(dict(house=2))+'\n')
            row=dict(house=1,episode_path='episode.json',episode_sha256=digest(root/'episode.json'))
            study=dict(train_index_path=str(train),train_index_sha256=digest(train))
            self.assertEqual(verify_inputs(root,study,row),root/'episode.json')
            with self.assertRaisesRegex(ValueError,'overlaps'):verify_inputs(root,study,dict(row,house=2))
            (root/'episode.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'episode changed'):verify_inputs(root,study,row)

    def test_small_batch_receipt_does_not_authorize_other_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'assets_batch_01').mkdir()
            manifest=root/'assets_batch_01/manifest.json';manifest.write_text('{}')
            receipt=dict(state='passed',benchmark_sha256='benchmark',asset_manifest_sha256=digest(manifest),
                         case_episode_sha256={'case_a':'episode_a'})
            (root/'staged_batch_01.json').write_text(json.dumps(receipt))
            study=dict(benchmark_sha256='benchmark',smoke_case_index=-1)
            row=dict(case_id='case_a',episode_sha256='episode_a',asset_stage_scope='batch_01')
            self.assertEqual(verify_assets(root,study,row,0)['state'],'passed')
            with self.assertRaisesRegex(ValueError,'outside'):
                verify_assets(root,study,dict(row,case_id='case_b'),1)


if __name__=='__main__':unittest.main()
