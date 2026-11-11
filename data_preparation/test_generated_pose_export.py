"""Exercise new recorded HDF5/video data through the existing pose input contract."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import imageio.v2 as imageio
import numpy as np

from data_preparation.export_benchmark_pose_labels import export_case
from data_preparation.verify_benchmark_expert import _save_h5
from data_preparation.witness_observations import calibration
from pose_learning.data import CAMERAS, load_examples, observed_points, state_vector


class GeneratedPoseExportTests(unittest.TestCase):
    def test_generated_recording_yields_current_base_pose_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'cases/episode_0001';folder.mkdir(parents=True)
            count=12;states=[];robot=[];frames=[];traces={'left':[],'right':[]}
            phases=[1,1,1,2,3,3,3,4,4,4,4,4]
            target=np.eye(4);target[:3,3]=[.6,.1,1.]
            for step in range(count):
                base=np.eye(4);base[0,3]=step*.01
                q=dict(head=[0.,0.],torso=[0.]*6,left_arm=[0.]*7,right_arm=[0.]*7,
                       left_gripper=[.04,.04] if step<5 else [.01,.01],
                       right_gripper=[.04,.04],base=[step*.01,0.,0.])
                robot.append(q);states.append(np.zeros(4))
                for arm in traces:
                    traces[arm].append(dict(robot_base_world=base,tcp_world=target,
                        selected_handle_world=target,benchmark_grasped=4<=step,
                        opening_fraction=0. if step<=6 else (step-6)/5,
                        gripper_joint_pos=q[arm+'_gripper']))
                frames.append(dict(simulation_time=step*.1,qvel={k:[0.]*len(v) for k,v in q.items()},
                    action_sensors={name:({'left_gripper':[.6,.1,1.,1.,0.,0.,0.]} if name=='ee_pose' else {})
                                    for name in ('joint_pos','joint_pos_rel','ee_pose','ee_twist')},
                    points={camera:dict(points=np.array([[.5,.5],[np.nan,np.nan]],np.float32),
                                        num_points=np.array([1])) for camera in CAMERAS},
                    camera_parameters={camera:calibration(np.eye(4),90.,(32,32,3)) for camera in CAMERAS}))
            for camera in CAMERAS:
                with imageio.get_writer(folder/(camera+'.mp4'),fps=10,macro_block_size=1) as writer:
                    for step in range(count):writer.append_data(np.full((32,32,3),step*10,np.uint8))
            row=dict(case_id='episode_0001',episode_sha256='episode',house=7)
            commands=[dict(action=dict(left_arm=[0.]*7)) for _ in range(count-1)]
            _save_h5(folder,dict(witness_format='benchmark_expert_witness_v2'),row,
                SimpleNamespace(model_dump=lambda:dict(house_index=7)),states,states,states,states,
                robot,traces,phases,['expert_prefix']*count,[False]*(count-1)+[True],commands,CAMERAS,
                frames=frames,scene_metadata=dict(task_description='Push the door open',policy_dt_ms=100.,
                    policy_phases=dict(GRASP_HANDLE=3,REACH_PRE_GRASP=1)))
            status=export_case(root,row)
            self.assertTrue(status['pose_label_eligible'])
            target_row=json.loads((folder/'pose_target.json').read_text())
            index=root/'targets.jsonl';index.write_text(json.dumps(target_row)+'\n')
            examples=load_examples(index,'test')
            self.assertEqual(len(examples),4)
            for example in examples:
                self.assertAlmostEqual(example['target'][0],.6-.01*example['frame'],places=5)
            with h5py.File(folder/'expert_trajectory.h5','r') as file:
                group=file['traj_0']
                np.testing.assert_allclose(observed_points(group,0),[[.5,.5,1.]]*3)
                self.assertEqual(state_vector(robot[0]).shape,(21,))


if __name__=='__main__':unittest.main()
