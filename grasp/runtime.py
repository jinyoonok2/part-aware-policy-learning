"""Small simulation/record helpers shared by the retained grasp code."""
import ast
import base64
import copy
import ctypes
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

class Record:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        self.__dict__.update(state.get('__dict__', state))
        self.__dict__.update(state.get('__pydantic_extra__') or {})

class ConfigReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('mujoco_thor.configs.', 'molmo_spaces.configs.')):
            return Record
        if module == 'pathlib' and name in ('Path', 'PosixPath', 'PurePosixPath'):
            import pathlib
            return getattr(pathlib, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np._core.multiarray, name)
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise ValueError(f'Unsupported config global: {module}.{name}')

def plain(value):
    if isinstance(value, Record):
        return plain(vars(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def write(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2) + '\n')
    temporary.replace(path)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def decode_row(row):
    raw = (row if isinstance(row, bytes) else row.tobytes()).decode('utf8').rstrip('\0')
    return json.loads(raw) if raw else {}

def pose_matrix(value):
    value = np.asarray(value)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    out[:3, 3] = value[:3]
    return out

def pose_error(actual, target):
    return {'position_m': float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            'orientation_deg': float(np.degrees(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()))}

def read_config(dataset, row):
    path = dataset / row['h5']
    assert digest(path) == row['h5_sha256'], 'Source HDF5 changed'
    with h5py.File(path, 'r') as h5:
        scene = decode_row(h5[row['trajectory']]['obs_scene'][()])
    config = plain(ConfigReader(io.BytesIO(base64.b64decode(scene['frozen_config']))).load())
    assert config['task_cls_str'].split('.')[-1] == 'DoorOpeningTask'
    assert config['robot_config']['name'] == 'rby1m'
    return config, scene

def physics_step(task, action):
    import mujoco
    robot = task.env.robots[0]
    robot.update_control({k: np.asarray(v, dtype=float) for k, v in action.items() if k in robot.controllers})
    for _ in range(task._n_ctrl_steps_per_policy):
        robot.compute_control()
        mujoco.mj_step(task.env.mj_model, task.env.mj_datas[0], task._n_sim_steps_per_ctrl)
    assert np.isfinite(task.env.mj_datas[0].qpos).all(), 'Nonfinite simulator state'

def integration_state(model, data):
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, value, spec)
    return value

def clone_task(task):
    """Clone physics and controllers without sharing mutable live simulation data.

    Branches deliberately skip sensors, caches, rendering and policy callbacks.
    They use precisely the robot/control/physics loop from BaseMujocoTask.step.
    Restricted to the pinned RBY1 door task and deterministic action controllers.
    """
    from molmo_spaces.env.data_views import Door
    from molmo_spaces.tasks.opening_tasks import DoorOpeningTask
    if not isinstance(task, DoorOpeningTask):
        raise TypeError('Grasp cloning supports DoorOpeningTask only')
    if task.env.n_batch != 1 or task.config.robot_config.action_noise_config.enabled:
        raise ValueError('Requires one environment and disabled action noise')
    source = task.env.robots[0]
    supported = {'JointRelPosController', 'JointPosController', 'TorsoHeightJointPosController'}
    if any(type(c).__name__ not in supported for c in source.controllers.values()):
        raise TypeError('Unsupported stateful controller; add an explicit clone contract first')
    data = copy.copy(task.env.mj_datas[0])  # Includes derived data, contacts and warm starts.
    robot = type(source)(data, source.exp_config)
    for name, controller in source.controllers.items():
        destination = robot.controllers[name]
        for key, value in vars(controller).items():
            if key != 'robot_move_group':
                setattr(destination, key, copy.deepcopy(value))
    robot._last_unnoised_cmd_joint_pos = copy.deepcopy(source._last_unnoised_cmd_joint_pos)
    branch = copy.copy(task)
    branch._env = SimpleNamespace(mj_model=task.env.mj_model, mj_datas=[data],
                                  robots=(robot,), n_batch=1)
    branch.door_object = Door(task.config.task_config.door_body_name, data)
    branch.current_door_joint_state = copy.deepcopy(task.current_door_joint_state)
    return branch

def select_allocated_egl_device():
    """Match EGL's CUDA ordinal, not Slurm's global GPU index or a DRM device."""
    import torch
    from mujoco.egl import egl_ext as EGL
    assert torch.cuda.device_count() == 1, 'Expected exactly one CUDA-visible allocated GPU'
    address = EGL.eglGetProcAddress(b'eglQueryDeviceAttribEXT')
    assert address, 'EGL CUDA device query is unavailable'
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(address)
    matches = []
    for index, device in enumerate(EGL.eglQueryDevicesEXT()):
        ordinal = ctypes.c_ssize_t()
        success = query(ctypes.cast(device, ctypes.c_void_p), 0x323A, ctypes.byref(ordinal))  # EGL_CUDA_DEVICE_NV
        if success and ordinal.value == torch.cuda.current_device(): matches.append(index)
    assert len(matches) == 1, f'Ambiguous EGL/CUDA mapping: {matches}'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(matches[0])

def digest_arrays(arrays):
    h = hashlib.sha256()
    for name, value in arrays:
        a = np.asarray(value)
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()

def physics_digest(model):
    names = ['body_mass', 'body_inertia', 'body_pos', 'body_quat', 'body_gravcomp',
             'geom_size', 'geom_pos', 'geom_quat', 'geom_type', 'geom_bodyid', 'geom_contype',
             'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin',
             'jnt_type', 'jnt_pos', 'jnt_axis', 'jnt_range', 'jnt_stiffness',
             'dof_damping', 'dof_frictionloss', 'dof_armature', 'actuator_gainprm',
             'actuator_biasprm', 'actuator_gear', 'actuator_ctrlrange', 'qpos0',
             'mesh_vert', 'mesh_face', 'cam_pos', 'cam_quat', 'cam_fovy']
    arrays = [(n, getattr(model, n)) for n in names]
    arrays += [('gravity', model.opt.gravity), ('timestep', model.opt.timestep)]
    return digest_arrays(arrays)

state = integration_state

def reset_state(model, data, reference):
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, reference, spec)
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, reference, spec)
    assert np.array_equal(state(model, data), reference)

def restore_point_arrays(value):
    if isinstance(value, dict):
        return {k: np.asarray(v, dtype=np.float32 if k == 'points' else np.int32)
                if k in ['points', 'num_points'] else restore_point_arrays(v)
                for k, v in value.items()}
    return value

