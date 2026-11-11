from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_preparation.export_policy_episodes import export, make_episode, resolve_scene


def config():
    return dict(camera_config=dict(cameras=[]), robot_config=dict(init_qpos=dict(base=[1.,2.,0.])),
        task_config=dict(door_body_name='door', robot_base_pose=[1.,2.,0.,1.,0.,0.,0.],
                         object_poses={'cup':[0.,0.,0.,1.,0.,0.,0.]}, added_objects={}))


def scene_file(assets, family='procthor-objaverse', house=3):
    path=assets/'scenes'/f'{family}-val'/f'val_{house}_ceiling.xml'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('<mujoco><worldbody><body name="door"/><body name="cup"/></worldbody></mujoco>')
    return path


class PolicyEpisodeExportChecks(unittest.TestCase):
    def test_initial_state_and_exposure_survive_without_expert_path_inputs(self):
        c=config();prior=deepcopy(c)
        row=dict(house=3,source_split='val')
        value=make_episode(row,c,dict(task_description='Push the door open',
            actions=['must not export'],target_pose=['must not export']),'procthor-objaverse')
        self.assertEqual(value['robot']['init_qpos'],c['robot_config']['init_qpos'])
        self.assertEqual(value['task']['robot_base_pose'],c['task_config']['robot_base_pose'])
        self.assertEqual(value['data_split'],'val')
        self.assertNotIn('source',value)
        self.assertNotIn('actions',value)
        self.assertNotIn('target_pose',json.dumps(value))
        self.assertEqual(c,prior)

    def test_scene_family_requires_one_exact_identity_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            assets=Path(temporary)
            row=dict(house=3,source_split='val')
            with self.assertRaises(ValueError):resolve_scene(row,config(),assets)
            scene_file(assets)
            selected,_=resolve_scene(row,config(),assets)
            self.assertEqual(selected['family'],'procthor-objaverse')
            scene_file(assets,'procthor-10k')
            with self.assertRaises(ValueError):resolve_scene(row,config(),assets)

    def test_missing_cases_block_runnable_full_manifest_and_remain_in_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);assets=root/'assets';scene_file(assets)
            source=root/'input.json'
            source.write_text(json.dumps(dict(cases=[dict(case_id=f'case{i}',house=3+i,source_split='val',
                previous_project_exposure=True,h5='a.h5',trajectory='traj_0') for i in range(2)])))
            args=Namespace(input=source,output=root/'output',dataset=root,assets=assets,
                cache=root/'cache',seed=42,limit=None,stage_scenes=False,allow_partial=False)
            with patch('data_preparation.export_policy_episodes.read_config',
                       return_value=(config(),dict(task_description='Push the door open'))):
                with self.assertRaisesRegex(RuntimeError,'No runnable'):
                    export(args)
            self.assertFalse((args.output/'policy_episode_manifest.json').exists())
            status=json.loads((args.output/'status.json').read_text())
            self.assertEqual(status['exported_cases'],1)
            self.assertEqual(status['pending_cases'],1)
            self.assertFalse(status['physical_grasp_retention_certified'])


if __name__=='__main__':unittest.main()
