"""Protect source-role separation, prior references and package extraction."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from data_preparation.prepare_train_test_dataset import (require_disjoint,safe_members,with_role,
    validate_reference_preservation,validate_trajectory_alignment,REQUIRED_STATE_ARRAYS,RGB_CAMERAS)


class TrainTestChecks(unittest.TestCase):
    def test_test_role_preserves_official_val_and_marks_exposure(self):
        source=dict(source_split='val',house=7,h5='a.h5',trajectory='traj_0')
        value=with_role(source,'test',Path('/tmp/test-data'),{'7':'expert replay already inspected'})
        self.assertEqual(value['source_split'],'val')
        self.assertEqual(value['project_role'],'test')
        self.assertTrue(value['previous_project_exposure'])
        self.assertNotIn('project_role',source)
        with self.assertRaises(ValueError):with_role(source,'train',Path('/tmp/train-data'))

    def test_house_or_content_overlap_rejects_different_path_spellings(self):
        train=[dict(house=1,h5_sha256='a',trajectory='traj_0')]
        with self.assertRaises(ValueError):require_disjoint(train,[dict(house=1,h5_sha256='b',trajectory='traj_9')])
        with self.assertRaises(ValueError):require_disjoint(train,[dict(house=2,h5_sha256='a',trajectory='traj_8')])
        with self.assertRaises(ValueError):require_disjoint(train,[dict(house=4,h5_sha256='c',trajectory='traj_0')],[4])
        require_disjoint(train,[dict(house=2,h5_sha256='b',trajectory='traj_0')])

    def archive(self,name='house_2/trace.h5',kind=tarfile.REGTYPE):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            entry=tarfile.TarInfo(name);entry.type=kind
            if kind==tarfile.REGTYPE:entry.size=3;archive.addfile(entry,io.BytesIO(b'abc'))
            else:entry.linkname='/outside';archive.addfile(entry)
        stream.seek(0)
        return tarfile.open(fileobj=stream,mode='r|')

    def test_valid_archive_is_readable_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.archive() as archive:receipt=safe_members(archive,Path(tmp),3)
            self.assertEqual((Path(tmp)/'house_2/trace.h5').read_bytes(),b'abc')
            self.assertEqual(receipt['unpacked_file_bytes'],3)
            with self.archive() as archive:
                with self.assertRaises(ValueError):safe_members(archive,Path(tmp),3)

    def test_archive_rejects_traversal_symlinks_and_budget_overrun(self):
        for name,kind,budget in [('../outside',tarfile.REGTYPE,9),('/outside',tarfile.REGTYPE,9),
                                 ('house_2/link',tarfile.SYMTYPE,9),('house_2/file',tarfile.REGTYPE,2)]:
            with self.subTest(name=name,kind=kind,budget=budget),tempfile.TemporaryDirectory() as tmp:
                with self.archive(name,kind) as archive:
                    with self.assertRaises(ValueError):safe_members(archive,Path(tmp),budget)

    def test_preservation_uses_source_identity_not_relocated_paths(self):
        source=dict(house=2,part=0,h5='old/trace.h5',trajectory='traj_0',h5_sha256='a',
                    interaction_event=dict(target_frame=5,measured_tcp_world_xyz_wxyz=[0,0,0,1,0,0,0]))
        moved=dict(source,h5='new/raw/trace.h5')
        validate_reference_preservation([moved],[source],'early pose')
        with self.assertRaises(ValueError):validate_reference_preservation([], [source],'trajectory')
        changed=json.loads(json.dumps(moved));changed['interaction_event']['target_frame']=6
        with self.assertRaises(ValueError):validate_reference_preservation([changed],[source],'early pose')

    def aligned_group(self, root):
        group={key:[None,None] for key in REQUIRED_STATE_ARRAYS}
        group['success']=[False,True]
        for camera in RGB_CAMERAS:
            filename=camera+'.mp4';(root/filename).write_bytes(b'test video placeholder')
            group['obs/sensor_data/'+camera]=filename.encode()
        return group

    def test_raw_video_alignment_is_independent_of_pose_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);group=self.aligned_group(root)
            # No pose event/label is provided: raw observations still get checked.
            value=validate_trajectory_alignment(group,root/'source.h5',root,
                video_reader=lambda *args,**kwargs:[0,1])
            self.assertTrue(value['video_observation_counts_checked'])
            self.assertTrue(value['required_state_observation_counts_checked'])
            self.assertEqual(set(value['videos']),set(RGB_CAMERAS))
            with self.assertRaises(ValueError):validate_trajectory_alignment(group,root/'source.h5',root,
                video_reader=lambda *args,**kwargs:[0])

    def test_missing_or_truncated_robot_state_is_an_integrity_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);group=self.aligned_group(root)
            group['obs/agent/qpos']=[None]
            with self.assertRaises(ValueError):validate_trajectory_alignment(group,root/'source.h5',root,
                video_reader=lambda *args,**kwargs:[0,1])
            del group['obs/agent/qpos']
            with self.assertRaises(ValueError):validate_trajectory_alignment(group,root/'source.h5',root,
                video_reader=lambda *args,**kwargs:[0,1])


if __name__=='__main__':unittest.main()
