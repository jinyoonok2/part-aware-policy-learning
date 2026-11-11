"""Read observed RGB/state/part points and separately stored future pose labels.

No simulator state, expert phase, future RGB, target pose or active-arm label is
fed into the model. The source phase was used by the data-preparation pipeline
to choose a supervision event, not as a predictor input.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('head_camera', 'wrist_camera_l', 'wrist_camera_r')
STATE_LAYOUT = (('head', (0, 1)), ('torso', (1, 2, 3)),
                ('left_arm', tuple(range(7))), ('right_arm', tuple(range(7))),
                ('left_gripper', (0,)), ('right_gripper', (0,)))
STATE_DIM = sum(len(indices) for _, indices in STATE_LAYOUT)
TARGET_FRAME = 'future initial-grasp TCP in current observation robot-base frame, xyz+wxyz'
POINT_RULE = 'mean of finite visible current-frame recorded door_handle segmentation points, normalized xy'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def decode(value):
    raw = value if isinstance(value, bytes) else np.asarray(value).tobytes()
    return json.loads(raw.rstrip(b'\0').decode('utf8'))


def state_vector(qpos):
    """Exclude world base coordinates to avoid memorizing house locations."""
    result = np.concatenate([np.asarray(qpos[name], dtype=np.float32)[list(indices)]
                             for name, indices in STATE_LAYOUT])
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError('Missing or invalid observed robot joint positions')
    return result


def point_vector(points):
    """Accept camera -> (normalized x,y,valid); absent points stay explicitly invalid."""
    result = np.zeros((3, 3), np.float32)
    for i, camera in enumerate(CAMERAS):
        raw = points.get(camera, [0, 0, 0])
        if isinstance(raw, dict):
            count = int(np.asarray(raw['num_points']).reshape(-1)[0])
            samples = np.asarray(raw['points'], np.float32)
            if samples.ndim != 2 or samples.shape[1] != 2 or not 0 <= count <= len(samples):
                raise ValueError('Invalid native part-point dimensions/count')
            visible = samples[:count]
            if count and (not np.isfinite(visible).all() or np.any(visible < 0) or np.any(visible > 1)):
                raise ValueError('Invalid normalized native part points')
            raw = [*visible.mean(0), 1] if count else [0,0,0]
        value = np.asarray(raw, np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid part point for {camera}')
        if value[2] > 0:
            if np.any(value[:2] < 0) or np.any(value[:2] > 1):
                raise ValueError('Visible part coordinates must lie in [0,1]')
            result[i] = [*value[:2], 1]
    return result


def observed_points(group, frame):
    points = {}
    for camera in CAMERAS:
        key = 'obs/extra/object_image_points/door_handle/' + camera
        if key not in group:
            continue
        count = int(np.asarray(group[key + '/num_points'][frame]).reshape(-1)[0])
        values = np.asarray(group[key + '/points'][frame], dtype=np.float32)
        if count < 0 or count > len(values):
            raise ValueError('Invalid source visible-point count')
        visible = values[:count]
        if len(visible):
            if not np.isfinite(visible).all() or np.any(visible < 0) or np.any(visible > 1):
                raise ValueError('Invalid source normalized point coordinates')
            points[camera] = [*visible.mean(0), 1]
    return point_vector(points)


def load_examples(index, role):
    """Validate the complete selected role before slicing for a smoke run."""
    if role not in ('train', 'test'):
        raise ValueError('Use exactly Train or Test')
    expected_source = {'train': 'train', 'test': 'val'}[role]
    index = Path(index).resolve()
    examples = []
    seen = set()
    roots = {}
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('source_split') != expected_source or row.get('project_role', role) != role:
            raise ValueError('Index contains a different source/project split')
        if row.get('quality_view') != 'initial_grasp_before_major_opening_v1':
            raise ValueError('Only the early-grasp quality view is accepted')
        root_name = str(row.get('dataset_root', index.parent))
        if root_name not in roots:
            roots[root_name] = Path(root_name).resolve()
        root = roots[root_name]
        # Prepared Train/Test views intentionally symlink raw/ to the audited
        # source dataset. Reject lexical traversal, but preserve that layout.
        relative = Path(row['h5'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Source HDF5 path must be relative without traversal')
        h5 = root / relative
        videos = {}
        for camera in CAMERAS:
            relative_video = Path(row['videos'][camera])
            if relative_video.is_absolute() or '..' in relative_video.parts:
                raise ValueError('Video source path must be relative without traversal')
            videos[camera] = str(root / relative_video)
        event = row['interaction_event']
        if not row['task_description'].startswith(('Push the door open','Pull the door open')):
            raise ValueError('Only push/pull door tasks are supported')
        if event['arm'] not in ('left', 'right'):
            raise ValueError('Invalid active-arm supervision')
        if row.get('selected_source_prior_max_opening_fraction', 1) > .1 + 1e-6:
            raise ValueError('Label was generated after prior major opening')
        width = float(np.abs(event['gripper_joint_positions']).sum())
        if not np.isfinite(width) or not 0 <= width <= .15:
            raise ValueError('Invalid measured sum of absolute finger displacements')
        for condition in row['condition_frames']:
            frame = int(condition['frame'])
            if not 0 <= frame < event['start_frame'] <= event['target_frame'] < row['observations']:
                raise ValueError('Conditioning must precede the labeled grasp event')
            pose = np.asarray(condition['target_tcp_in_observation_base_xyz_wxyz'], np.float32)
            if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > .02:
                raise ValueError('Invalid target TCP pose')
            key = (row['h5_sha256'], row['trajectory'], frame)
            if key in seen:
                raise ValueError('Duplicate conditioning observation')
            seen.add(key)
            examples.append(dict(h5=str(h5), trajectory=row['trajectory'], frame=frame,
                house=int(row['house']), target=pose.tolist(), arm=int(event['arm']=='right'),
                gripper_width_m=width, videos=videos,
                is_pull=float(row['task_description'].startswith('Pull')), source_split=expected_source,
                source_h5_sha256=row['h5_sha256'], source_observations=int(row['observations'])))
    if not examples:
        raise ValueError('Empty pose index')
    return examples


class CachedPoseDataset:
    """Frozen vision features plus Train-fitted robot-state normalization."""
    def __init__(self, directory, role, normalization):
        import torch
        self.torch = torch
        directory = Path(directory)
        self.metadata = json.loads((directory/'metadata.json').read_text())
        if self.metadata['role'] != role or self.metadata['target_frame'] != TARGET_FRAME:
            raise ValueError('Feature cache role or pose-frame mismatch')
        if self.metadata.get('complete') is not True:
            raise ValueError('Feature cache was not completed')
        if normalization.get('fitted_on') != 'train only':
            raise ValueError('Robot-state normalization must come from Train only')
        for name in ('features.npy', 'arrays.npz', 'examples.json'):
            if digest(directory/name) != self.metadata['files_sha256'][name]:
                raise ValueError('Feature cache integrity mismatch: '+name)
        self.visual = np.load(directory/'features.npy', mmap_mode='r', allow_pickle=False)
        with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
            self.arrays = {name: arrays[name] for name in arrays.files}
        n = len(self.visual)
        if self.visual.shape != (n, 3, 2, 768) or any(len(x) != n for x in self.arrays.values()):
            raise ValueError('Feature/label alignment mismatch')
        expected=dict(state=(n,STATE_DIM),points=(n,3,3),target=(n,7),arm=(n,),gripper_width_m=(n,),is_pull=(n,))
        if set(self.arrays)!=set(expected) or any(self.arrays[name].shape!=shape for name,shape in expected.items()):
            raise ValueError('Feature cache array schema mismatch')
        self.normalization = normalization

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, index):
        item = {name:self.torch.from_numpy(np.array(value[index], copy=True)) for name, value in self.arrays.items()}
        item['visual'] = self.torch.from_numpy(np.array(self.visual[index], dtype=np.float32))
        item['state'] = ((item['state']-self.torch.tensor(self.normalization['mean']))/
                         self.torch.tensor(self.normalization['std'])).clamp(-10, 10)
        return item


def fit_normalization(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['role'] != 'train' or not metadata['complete']:
        raise ValueError('Normalization can only be fitted on completed Train features')
    if digest(directory/'arrays.npz') != metadata['files_sha256']['arrays.npz']:
        raise ValueError('Train arrays changed before normalization fitting')
    with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
        state = arrays['state'].astype(np.float64)
    if state.ndim!=2 or state.shape[1]!=STATE_DIM or not len(state) or not np.isfinite(state).all():
        raise ValueError('Invalid Train robot-state normalization input')
    return dict(mean=state.mean(0).tolist(), std=np.maximum(state.std(0), .01).tolist(), fitted_on='train only')
