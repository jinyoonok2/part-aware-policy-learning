"""Current-frame observations for newly generated benchmark expert records."""
import json
import numpy as np


def calibration(camera_to_world, fovy_deg, image_shape):
    height, width = image_shape[:2]
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not 0 < fovy_deg < 180:
        raise ValueError('Invalid live camera pose or field of view')
    focal = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    intrinsic = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    # MuJoCo/OpenGL looks along -Z with +Y up; CV uses +Z and +Y down.
    extrinsic = np.diag([1., -1., -1.]) @ np.linalg.inv(pose)[:3]
    return dict(cam2world_gl=pose.copy(), extrinsic_cv=extrinsic, intrinsic_cv=intrinsic)


def current_handle_points(observation, camera):
    values = observation['object_image_points']['door_handle'][camera]
    points = np.asarray(values['points'], dtype=np.float32)
    count = int(np.asarray(values['num_points']).reshape(-1)[0])
    if points.ndim != 2 or points.shape[1] != 2 or not 0 <= count <= len(points):
        raise ValueError('Invalid native current-frame point dimensions')
    visible = points[:count]
    if not np.isfinite(visible).all() or np.any(visible < 0) or np.any(visible > 1):
        raise ValueError('Invalid native current-frame normalized handle points')
    return dict(points=points.copy(), num_points=np.array([count], dtype=np.int32))


def capture(task, observation, cameras):
    obs = observation[0] if isinstance(observation, (list, tuple)) else observation
    frame = dict(simulation_time=float(task.env.mj_datas[0].time),
        qvel={key: value.copy() for key, value in task.env.robots[0].robot_view.get_qvel_dict().items()},
        points={}, camera_parameters={}, action_sensors={})
    for camera in cameras:
        image = np.asarray(obs[camera])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Recorded camera must supply current uint8 RGB')
        live = task.env.camera_manager.registry[camera]
        frame['camera_parameters'][camera] = calibration(live.get_pose(), float(live.fov), image.shape)
        frame['points'][camera] = current_handle_points(obs, camera)
    for name in ('joint_pos', 'joint_pos_rel', 'ee_pose', 'ee_twist'):
        value = obs['actions/'+name]
        if not isinstance(value, dict):
            raw = value if isinstance(value, bytes) else np.asarray(value).tobytes()
            value = json.loads(raw.rstrip(b'\0').decode())
        frame['action_sensors'][name] = value
    return frame


def save(group, frames, cameras, encoded):
    if len(frames) != len(group['success']):
        raise ValueError('Current observations and recorded states are not aligned')
    times = np.asarray([frame['simulation_time'] for frame in frames])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Observation times must increase strictly')
    group.create_dataset('obs/time', data=times)
    group.create_dataset('obs/agent/qvel', data=encoded([frame['qvel'] for frame in frames]))
    for name in ('joint_pos', 'joint_pos_rel', 'ee_pose', 'ee_twist'):
        group.create_dataset('actions/'+name, data=encoded([frame['action_sensors'][name] for frame in frames]))
    for camera in cameras:
        for key in ('points', 'num_points'):
            values = np.asarray([frame['points'][camera][key] for frame in frames])
            group.create_dataset('obs/extra/object_image_points/door_handle/'+camera+'/'+key,
                                 data=values, compression='gzip')
        for key in ('cam2world_gl', 'extrinsic_cv', 'intrinsic_cv'):
            values = np.asarray([frame['camera_parameters'][camera][key] for frame in frames])
            group.create_dataset('obs/sensor_param/'+camera+'/'+key, data=values, compression='gzip')
    group['obs/sensor_param'].attrs['convention'] = 'OpenGL cam2world; CV extrinsic flips OpenGL Y/Z; intrinsics use actual rendered RGB dimensions'
