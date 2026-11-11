"""Measure noise-disabled physical replay of an existing source expert episode.

Recorded absolute commands drive the simulator. Source poses are read only for
initial alignment and tracking diagnostics; no state is set along a trajectory.
The newly executed force/retention witness is distinct from original source GT.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.verify_benchmark_expert import (
    canonical_hash, checked_existing, continuation_count, digest,
    initial_alignment, plain, write, _save_h5)


def command_frames(actions):
    """Source frame zero is empty padding; action[k] produces observation[k]."""
    if not actions or actions[0]:
        raise ValueError('Expected empty commanded_action at source frame zero')
    for frame, action in enumerate(actions[1:], 1):
        if not isinstance(action, dict):
            raise ValueError('Recorded command must be a mapping')
        if not action or action.get('done', False):
            break
        yield frame, action


def bounded_replay_plan(actions, horizon=400, hold=10):
    """Return the prefix and budget-limited uniform last-command continuation."""
    if type(horizon) is not int or horizon < 1 or type(hold) is not int or not 0 <= hold <= 20:
        raise ValueError('Invalid replay or continuation budget')
    commands = list(command_frames(actions))
    prefix = commands[:horizon]
    # Continue only after the complete available source prefix, never after
    # silently truncating a longer trajectory to make room for extra holding.
    count = continuation_count(len(prefix), horizon, hold) if len(prefix) == len(commands) and prefix else 0
    return prefix, count, len(prefix) < len(commands)


def configure_source_robot(cfg, source):
    from molmo_spaces.configs.robot_configs import ActionNoiseConfig
    robot = source['robot_config']
    for name in ('robot_xml_path', 'gravcomp', 'K_stiffness', 'K_damping', 'use_holo_base'):
        if name in robot:
            setattr(cfg.robot_config, name, Path(robot[name]) if name == 'robot_xml_path' else robot[name])
    for name, value in robot['command_mode'].items():
        if cfg.robot_config.command_mode.get(name) != value:
            raise ValueError(f'Source absolute controller mismatch for {name}')
    noise = dict(robot['action_noise_config'], enabled=False)
    cfg.robot_config.action_noise_config = ActionNoiseConfig(**noise)
    return noise


def execute_command(task, action, step, substeps):
    """Use the native robot/control/env stepping loop with source absolute commands."""
    import numpy as np
    robot = task.env.robots[0]
    data = task.env.mj_datas[0]
    unsupported = set(action) - set(robot.controllers) - {'done'}
    ignored = {}
    if 'head' in unsupported:
        modes = getattr(getattr(getattr(task, 'config', None), 'robot_config', None), 'command_mode', {})
        # Released experts emit a head target even when head actuation is
        # explicitly disabled. Preserve/log that source command without
        # inventing a controller or rejecting an otherwise valid replay.
        if 'head' in modes and modes['head'] is None:
            ignored['head'] = copy.deepcopy(action['head'])
            unsupported.remove('head')
    if unsupported:
        raise ValueError('Recorded command has unsupported groups: ' + ', '.join(sorted(unsupported)))
    robot.update_control({key: np.asarray(value, dtype=float) for key, value in action.items()
                          if key in robot.controllers})
    targets = {name: controller.target_pos.copy() for name, controller in robot.controllers.items()
               if hasattr(controller, 'target_pos')}
    for substep in range(task._n_ctrl_steps_per_policy):
        robot.compute_control()
        substeps.append(dict(policy_step=step, control_substep=substep,
                             simulation_time=float(data.time), n_sim_steps=task._n_sim_steps_per_ctrl,
                             actuator_controls=data.ctrl.copy(), ignored_nonactuated_commands=ignored))
        task.env.step(task._n_sim_steps_per_ctrl)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError('Nonfinite replay physics')
    return targets


def verify_prepared(run, study, row):
    from data_preparation.prepare_existing_test_verification import require_frozen
    require_frozen(run, study)
    if not study.get('stage_complete') or not study.get('worker_source_frozen'):
        raise ValueError('Complete frozen source staging is required before replay')
    folder = run / 'cases' / row['case_id']
    receipt = json.loads((folder / 'staging.json').read_text())
    if receipt.get('state') != 'passed' or receipt['episode_sha256'] != row['episode_sha256']:
        raise ValueError('Per-case staging receipt disagrees with frozen EpisodeSpec')
    if digest(run / row['episode_path']) != row['episode_sha256']:
        raise ValueError('Frozen EpisodeSpec changed')
    if digest(folder / 'decoded_config.json') != row['source_config_sha256']:
        raise ValueError('Source robot/task configuration changed')
    if digest(receipt['scene_identity']['xml']) != receipt['scene_identity']['xml_sha256']:
        raise ValueError('Matched source scene changed')
    return folder


def worker(args):
    import h5py
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import torch
    from grasp.contacts import ContactProbe
    from grasp.metrics import FINAL_POLICY, retained_grasp_opening
    from grasp.runtime import (decode_row, integration_state, physics_digest, pose_error,
                               pose_matrix, select_allocated_egl_device)
    from molmo_spaces.configs.robot_configs import ActionNoiseConfig
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    from olmo.eval.configure_molmo_spaces import RBY1RecordedTrajectoryEvalConfig

    run = args.run
    study = json.loads((run / 'study.json').read_text())
    row = study['cases'][args.index]
    case = verify_prepared(run, study, row)
    if checked_existing(case, row['episode_sha256']):
        print('Existing completed replay verified: ' + row['case_id'], flush=True)
        return
    if (case / 'worker_started.json').exists():
        raise RuntimeError('Preserve the previous incomplete replay; use an explicit fresh attempt')
    source = json.loads((case / 'decoded_config.json').read_text())
    scene = json.loads((case / 'decoded_scene.json').read_text())
    path = Path(row['dataset_root']) / row['h5']
    if digest(path) != row['h5_sha256']:
        raise ValueError('Original source HDF5 changed')
    with h5py.File(path, 'r') as data:
        group = data[row['trajectory']]
        actions = [decode_row(value) for value in group['actions/commanded_action']]
        base = np.asarray(group['obs/extra/robot_base_pose'][:])
        source_tcp = {arm: np.asarray(group[f'obs/extra/{arm}_tcp_pose'][:]) for arm in ('left', 'right')}
        source_success = np.asarray(group['success'][:], dtype=bool)
    if len(actions) != len(base) or any(len(values) != len(actions) for values in source_tcp.values()):
        raise ValueError('Source state/action arrays are not aligned')
    if float(scene['policy_dt_ms']) != 100.:
        raise ValueError('Source commands may not be resampled')
    horizon = int(study['horizon'])
    prefix, requested_hold, source_truncated = bounded_replay_plan(actions, horizon, int(study['continuation_steps']))
    policy = study['metric_policy']
    if canonical_hash(policy) != canonical_hash(FINAL_POLICY):
        raise ValueError('Frozen retention metric differs from imported implementation')
    select_allocated_egl_device()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Replay rendering requires an allocated CUDA device')
    seed = int(study['audit_seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    spec = EpisodeSpec.model_validate_json((run / row['episode_path']).read_text())
    if spec.data_split != 'val' or spec.house_index != row['house']:
        raise ValueError('Source split/house differs from EpisodeSpec')
    write(case / 'worker_started.json', dict(pid=os.getpid(), source_h5_sha256=row['h5_sha256'],
                                            episode_sha256=row['episode_sha256']))
    sampler = None
    writers = {}
    traces = {'left': [], 'right': []}
    states, qpos, qvel, controls, robot_states = [], [], [], [], []
    phases, stages, successes, commands, substeps = [], [], [], [], []
    tracking = []
    began = time.perf_counter()
    try:
        write(case / 'status.json', dict(state='loading_source_replay_scene', case_id=row['case_id']))
        cfg = RBY1RecordedTrajectoryEvalConfig()
        cfg.num_workers = 1; cfg.use_wandb = False; cfg.output_dir = case
        cfg.task_horizon = horizon; cfg.terminate_upon_success = False
        cfg.policy_dt_ms = 100.; cfg.ctrl_dt_ms = 20.; cfg.sim_dt_ms = 4.
        settings = configure_source_robot(cfg, source)
        sampler = JsonEvalTaskSampler(cfg, spec)
        cfg.robot_config.action_noise_config = ActionNoiseConfig(**settings)
        task = sampler.sample_task(house_index=row['house'])
        if task is None:
            raise RuntimeError('Source task sampler did not produce a task')
        robot = task.env.robots[0]
        robot.exp_config.robot_config.action_noise_config = ActionNoiseConfig(**settings)
        model, data, view = task.env.mj_model, task.env.mj_datas[0], robot.robot_view
        probes = {arm: ContactProbe(task, arm) for arm in traces}
        live_qpos = {name: view.get_move_group(name).joint_pos.copy() for name in spec.robot.init_qpos}
        object_poses = {}
        for name in spec.scene_modifications.object_poses:
            body = model.body(name).id
            matrix = np.eye(4); matrix[:3, 3] = data.xpos[body]
            matrix[:3, :3] = data.xmat[body].reshape(3, 3)
            object_poses[name] = matrix
        alignment = initial_alignment(spec.model_dump(), live_qpos, view.base.pose.copy(),
            task.door_object.get_joint_position(task.door_object.get_hinge_joint_index()), object_poses)
        alignment['source_initial_tcp'] = {arm: pose_error(view.get_move_group(arm + '_gripper').leaf_frame_to_world,
            pose_matrix(base[0]) @ pose_matrix(source_tcp[arm][0])) for arm in traces}
        alignment['passed'] = bool(alignment['passed'] and all(
            value['position_m'] <= .02 and value['orientation_deg'] <= 5.
            for value in alignment['source_initial_tcp'].values()))
        write(case / 'initial_alignment.json', alignment)
        if not alignment['passed']:
            raise RuntimeError('Source initial-state/TCP alignment failed; feasibility remains unknown')
        write(case / 'resolved_config.json', dict(native_config=cfg.model_dump(), episode_spec=spec.model_dump(),
            decoded_source_config=source, expert_source='Original released expert absolute commands',
            original_source_noise=source['robot_config']['action_noise_config'], replay_noise=settings,
            original_source_rng_available=False, exact_historical_noise_replay_claim=False,
            full_controller_equivalence_established=False,
            command_interpretation='Original absolute joint/base commands; direct native physical control loop. No trajectory teleportation.'))
        np.savez_compressed(case / 'initial_state.npz', integration_state=integration_state(model, data),
                            qpos=data.qpos.copy(), qvel=data.qvel.copy(), actuator_control=data.ctrl.copy(),
                            base_world=view.base.pose.copy())
        # Native scenes are compiled from MjSpec rather than loaded XML, so
        # mj_saveLastXML has no source XML to serialize. Preserve the complete
        # compiled model as a reloadable native MJB instead.
        mujoco.mj_saveModel(model, str(case / 'compiled_initial_scene.mjb'), None)
        cameras = [name for name in ('head_camera', 'wrist_camera_l', 'wrist_camera_r')
                   if name in {camera.name for camera in cfg.camera_config.cameras}]
        if 'head_camera' not in cameras:
            raise ValueError('Source replay has no head camera')
        for camera in cameras:
            writers[camera] = imageio.get_writer(case / (camera + '.mp4'), fps=10, macro_block_size=1)

        def sample(step, stage, source_frame=None):
            for arm, probe in probes.items():
                entry = probe.measure(task, arm, np.eye(4)); entry.pop('tcp_error', None)
                entry.update(step=step, stage=stage, source_frame=source_frame,
                             robot_base_world=view.base.pose.copy())
                traces[arm].append(entry)
            states.append(integration_state(model, data)); qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy()); controls.append(data.ctrl.copy())
            robot_states.append({name: view.get_move_group(name).joint_pos.copy() for name in spec.robot.init_qpos})
            phases.append(-1); stages.append(stage); successes.append(bool(task.judge_success()))
            for camera, writer in writers.items():
                writer.append_data(task.env.render_rgb_frame(camera))

        sample(0, 'initial', 0)
        initial_opening = traces['left'][0]['opening_fraction']
        initial_eligible = initial_opening <= policy['max_opening_at_grasp'] and not successes[0]
        source_steps = hold_steps = 0
        if initial_eligible:
            for frame, action in prefix:
                step = len(commands) + 1
                targets = execute_command(task, copy.deepcopy(action), step, substeps)
                commands.append(dict(step=step, stage='source_command_replay', source_frame=frame,
                                     action=copy.deepcopy(action), applied_controller_targets=targets))
                sample(step, 'source_command_replay', frame)
                tracking.append(dict(source_frame=frame, recorded_success=bool(source_success[frame]),
                    per_arm_tcp={arm: pose_error(traces[arm][-1]['tcp_world'],
                        pose_matrix(base[frame]) @ pose_matrix(source_tcp[arm][frame])) for arm in traces}))
                if step % 25 == 0:
                    write(case / 'status.json', dict(state='replaying_source_expert', case_id=row['case_id'],
                                                     executed_actions=step))
            source_steps = len(commands)
        threshold = float(task.config.task_config.door_openness_threshold)
        prefix_metric = {arm: retained_grasp_opening(values, threshold, policy) for arm, values in traces.items()}
        # The same declared hold protocol applies to all completed source
        # prefixes. Prefix/continuation retention is recorded separately.
        if initial_eligible and prefix:
            for _ in range(requested_hold):
                step = len(commands) + 1
                action = copy.deepcopy(prefix[-1][1])
                targets = execute_command(task, action, step, substeps)
                commands.append(dict(step=step, stage='last_command_hold', source_frame=None,
                                     action=action, applied_controller_targets=targets))
                sample(step, 'last_command_hold'); hold_steps += 1
        for writer in writers.values():
            writer.close()
        writers = {}
        measured = {arm: retained_grasp_opening(values, threshold, policy) for arm, values in traces.items()}
        passed = bool(initial_eligible and any(value['retained_grasp_opening_success'] for value in measured.values()))
        poses = {}
        for arm, value in measured.items():
            if value['retained_grasp_opening_success']:
                onset = next(entry for entry in traces[arm] if entry['step'] == value['grasp_onset_step'])
                poses[arm] = dict(grasp_onset_step=onset['step'], tcp_world=onset['tcp_world'],
                    tcp_in_robot_base=np.linalg.inv(np.asarray(onset['robot_base_world'])) @ np.asarray(onset['tcp_world']),
                    tcp_in_handle=np.linalg.inv(np.asarray(onset['selected_handle_world'])) @ np.asarray(onset['tcp_world']),
                    gripper_joint_pos=onset['gripper_joint_pos'], origin='New measured replay witness, not original source force ground truth')
        write(case / 'grasp_poses.json', poses)
        for arm, values in traces.items():
            write(case / (arm + '_force_trace.json'), values)
        for name, values in [('executed_commands.jsonl', commands), ('applied_control_substeps.jsonl', substeps)]:
            with (case / name).open('w') as stream:
                for entry in values:
                    stream.write(json.dumps(entry, default=plain, allow_nan=False) + '\n')
        write(case / 'source_tracking.json', dict(source_h5_sha256=row['h5_sha256'],
            source_trajectory=row['trajectory'], tracking=tracking))
        _save_h5(case, study, row, spec, states, qpos, qvel, controls, robot_states, traces,
                 phases, stages, successes, commands, cameras)
        with h5py.File(case / 'expert_trajectory.h5', 'r+') as file:
            file.attrs['origin'] = 'new measured noise-disabled physical replay of original released expert commands'
            file.attrs['source_h5_sha256'] = row['h5_sha256']
            file.attrs['source_trajectory'] = row['trajectory']
            file.attrs['witness_mode'] = study['replay_mode']
        if digest(path) != row['h5_sha256']:
            raise ValueError('Original source HDF5 changed during read-only replay')
        filenames = ['initial_state.npz', 'expert_trajectory.h5', 'left_force_trace.json',
            'right_force_trace.json', 'executed_commands.jsonl', 'applied_control_substeps.jsonl',
            'head_camera.mp4', 'grasp_poses.json', 'initial_alignment.json', 'resolved_config.json',
            'compiled_initial_scene.mjb', 'source_tracking.json']
        filenames += [camera + '.mp4' for camera in cameras if camera != 'head_camera']
        summary = dict(state='complete', case_id=row['case_id'], original_episode_index=None,
            house=row['house'], source_split='val', project_role='test', episode_sha256=row['episode_sha256'],
            origin='new_measured_replay_of_released_expert_commands', source_h5_sha256=row['h5_sha256'],
            source_trajectory=row['trajectory'], source_observations=len(actions), source_recorded_success=bool(source_success.any()),
            replay_mode=study['replay_mode'], original_exact_noise_replay_claim=False,
            compiled_scene_artifact=dict(path='compiled_initial_scene.mjb', format='MuJoCo binary model (MJB)'),
            nonactuated_head_commands=sum('head' in entry['action'] and 'head' not in robot.controllers for entry in commands),
            exact_start_verified=True, initial_alignment_passed=True, initial_eligible=bool(initial_eligible),
            initial_opening_fraction=initial_opening, horizon=horizon, executed_actions=len(commands),
            prefix_steps=source_steps, continuation_steps=hold_steps, source_prefix_truncated=source_truncated,
            source_prefix_retention=prefix_metric, retained_grasp_by_arm=measured,
            ordinary_opening_ever=bool(any(successes)), threshold=threshold,
            metric_version=policy['metric_version'], metric_policy_sha256=canonical_hash(policy),
            retained_grasp_witness_passed=passed, qualified_for_model_test=False, final_model_test_ready=False,
            feasibility_status='measured_replay_witness' if passed else 'not_demonstrated_by_this_replay',
            intrinsic_task_infeasible=False, full_controller_equivalence_established=False,
            model_inference_used=False, training_started=False, baseline_outcomes_used=False,
            physics_sha256=physics_digest(model), elapsed_seconds=time.perf_counter() - began,
            artifact_sha256={name: digest(case / name) for name in filenames})
        write(case / 'summary.json', summary); write(case / 'status.json', summary)
        print(json.dumps({key: summary[key] for key in ('case_id', 'state', 'executed_actions', 'retained_grasp_witness_passed')}, indent=2), flush=True)
    except Exception:
        for arm, values in traces.items():
            if values:
                write(case / (arm + '_partial_force_trace.json'), values)
        if states:
            np.savez_compressed(case / 'partial_trajectory.npz', integration_states=np.asarray(states),
                                 qpos=np.asarray(qpos), qvel=np.asarray(qvel), actuator_controls=np.asarray(controls))
        for name, values in [('partial_executed_commands.jsonl', commands), ('partial_control_substeps.jsonl', substeps)]:
            if values:
                with (case / name).open('w') as stream:
                    for entry in values:
                        stream.write(json.dumps(entry, default=plain, allow_nan=False) + '\n')
        raise
    finally:
        for writer in writers.values():
            writer.close()
        if sampler is not None:
            sampler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    study = json.loads((args.run / 'study.json').read_text())
    if not 0 <= args.index < len(study['cases']):
        parser.error('--index is outside the frozen existing Test inventory')
    try:
        worker(args)
    except Exception:
        row = study['cases'][args.index]
        case = args.run / 'cases' / row['case_id']
        value = dict(state='runtime_or_compatibility_error', case_id=row['case_id'],
            error=traceback.format_exc(), feasibility_status='unknown', intrinsic_task_infeasible=False,
            model_inference_used=False, training_started=False)
        write(case / 'worker_error.json', value); write(case / 'status.json', value)
        raise
    finally:
        reporter = args.run / 'source/report_benchmark_expert_verification.py'
        if reporter.is_file():
            subprocess.run(['/usr/bin/python3', str(reporter), '--run', str(args.run)], check=False)


if __name__ == '__main__':
    main()
