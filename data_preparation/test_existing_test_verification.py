import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_preparation.existing_test_worker import bounded_replay_plan, command_frames, execute_command, verify_prepared
from data_preparation.prepare_existing_test_verification import prepare, source_cases
from data_preparation.prepare_test_set import canonical, digest, write
from data_preparation.report_benchmark_expert_verification import REQUIRED, expected_protocol, inspect_case
from grasp.metrics import FINAL_POLICY


class ExistingTestVerificationChecks(unittest.TestCase):
    def test_every_source_case_is_retained_without_pose_eligibility_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'inventory.json'
            rows = [dict(case_id=f'case_{i}', house=i, source_split='val', project_role='test',
                         pose_eligible=False, h5='unused.h5', trajectory='traj_0', h5_sha256='a'*64)
                    for i in range(2)]
            write(manifest, dict(cases=rows))
            args = Namespace(manifest=manifest, run=root/'run', dataset=root,
                             project=Path(__file__).resolve().parents[1])
            with patch('data_preparation.prepare_existing_test_verification.read_config',
                       return_value=({'robot_config': {}}, {'policy_dt_ms': 100.})), \
                 patch('data_preparation.prepare_existing_test_verification.freeze_packages', return_value={}):
                prepare(args)
            study = json.loads((args.run/'study.json').read_text())
            self.assertEqual([row['case_id'] for row in study['cases']], ['case_0', 'case_1'])
            self.assertEqual(study['existing_expert_test_trajectories'], 2)
            self.assertFalse(study['stage_complete'])
            self.assertTrue(all(row['original_episode_index'] is None for row in study['cases']))
            self.assertEqual(json.loads(manifest.read_text())['cases'], rows)

    def test_duplicate_or_wrong_source_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'inventory.json'
            base = dict(case_id='one', house=1, source_split='val', project_role='test')
            write(path, dict(cases=[base,base]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                source_cases(path)
            write(path, dict(cases=[dict(base,source_split='train')]))
            with self.assertRaisesRegex(ValueError, 'official Val'):
                source_cases(path)

    def test_frame_zero_padding_and_done_do_not_execute_as_controls(self):
        actions = [{}, {'left_arm': [1]}, {'left_arm': [2]}, {'done': True}, {'left_arm': [999]}]
        self.assertEqual([frame for frame,_ in command_frames(actions)], [1,2])
        with self.assertRaisesRegex(ValueError, 'frame zero'):
            list(command_frames([{'left_arm':[1]}]))

    def test_source_prefix_and_continuation_share_one_400_action_budget(self):
        actions = [{}]+[{'left_arm':[float(i)]} for i in range(395)]
        prefix, hold, truncated = bounded_replay_plan(actions)
        self.assertEqual((len(prefix),hold,truncated),(395,5,False))
        actions += [{'left_arm':[400.]}]*20
        prefix,hold,truncated = bounded_replay_plan(actions)
        self.assertEqual((len(prefix),hold,truncated),(400,0,True))
        self.assertEqual(bounded_replay_plan([{}]),([],0,False))

    def test_execution_advances_native_control_substeps_and_rejects_unknown_groups(self):
        class Robot:
            def __init__(self):
                self.controllers={'left_arm':SimpleNamespace(target_pos=np.zeros(1))}
                self.command=None
            def update_control(self,values):
                self.command=values
                self.controllers['left_arm'].target_pos=values['left_arm'].copy()
            def compute_control(self):
                data.ctrl[:] = self.controllers['left_arm'].target_pos
        data=SimpleNamespace(ctrl=np.zeros(1),qpos=np.zeros(1),qvel=np.zeros(1),time=0.)
        robot=Robot()
        def step(count):
            data.time+=.004*count
            data.qpos+=data.ctrl*.004*count
        task=SimpleNamespace(env=SimpleNamespace(robots=[robot],mj_datas=[data],step=step),
                             config=SimpleNamespace(robot_config=SimpleNamespace(command_mode={'head': None})),
                             _n_ctrl_steps_per_policy=5,_n_sim_steps_per_ctrl=5)
        log=[]
        result=execute_command(task,{'left_arm':[2.], 'head':[.1,.2]},1,log)
        self.assertEqual(len(log),5)
        self.assertAlmostEqual(data.time,.1)
        self.assertAlmostEqual(data.qpos[0],.2)
        np.testing.assert_array_equal(result['left_arm'],[2.])
        self.assertEqual(log[0]['ignored_nonactuated_commands'], {'head':[.1,.2]})
        self.assertNotIn('head', robot.command)
        task.config.robot_config.command_mode['head']='joint_position'
        with self.assertRaisesRegex(ValueError,'unsupported'):
            execute_command(task,{'left_arm':[2.], 'head':[.1,.2]},2,log)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            execute_command(task,{'future_pose':[1.]},2,log)

    def test_source_replay_bundle_is_accepted_by_generic_qualification_reporter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);case=root/'cases/source_case';case.mkdir(parents=True)
            write(case/'episode.json',{'initial_state':'frozen'})
            row=dict(case_id='source_case',original_episode_index=None,house=7,
                     episode_path='cases/source_case/episode.json',episode_sha256=digest(case/'episode.json'))
            study=dict(metric_policy=FINAL_POLICY,metric_policy_sha256=canonical(FINAL_POLICY),horizon=400)
            for name in REQUIRED:
                (case/name).write_text('{}')
            write(case/'initial_alignment.json',dict(passed=True))
            write(case/'grasp_poses.json',dict(left=dict(origin='new measured replay')))
            summary=dict(row,state='complete',metric_version=FINAL_POLICY['metric_version'],
                metric_policy_sha256=canonical(FINAL_POLICY),horizon=400,executed_actions=150,
                model_inference_used=False,training_started=False,retained_grasp_witness_passed=True,
                exact_start_verified=True,initial_alignment_passed=True,
                origin='new_measured_replay_of_released_expert_commands',
                artifact_sha256={name:digest(case/name) for name in REQUIRED})
            write(case/'summary.json',summary)
            result=inspect_case(root,row,expected_protocol(study),{})
            self.assertEqual(result['classification'],'completed_pass')
            (case/'head_camera.mp4').write_text('changed')
            self.assertEqual(inspect_case(root,row,expected_protocol(study),{})['classification'],'infrastructure_unknown')


if __name__ == '__main__':
    unittest.main()
