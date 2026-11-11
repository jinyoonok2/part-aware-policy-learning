"""Matched simulator evaluation of frozen MolmoBot and pose corrections.

Consumes explicit EpisodeSpec paths. The optional oracle_pose diagnostic reads
an explicitly supplied expert reference pose; other methods never use it. The
strict grasp metric reads simulator contacts for measurement; the intervention
uses the predicted TCP pose, model closing intent, robot kinematics and contact
feedback. This command does not prepare or certify the Test dataset.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np


CAMERAS = ('head_camera', 'wrist_camera_l', 'wrist_camera_r')


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, default=plain, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(value):
    """Include array dtype and NaN padding in observation identities."""
    def walk(item):
        if isinstance(item, np.ndarray):
            return {'shape': item.shape, 'dtype': str(item.dtype),
                    'sha256': hashlib.sha256(item.tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {str(k): walk(v) for k, v in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [walk(v) for v in item]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return hashlib.sha256(json.dumps(walk(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def load_cases(path):
    """Require explicit materialized starts; refuse arbitrary trajectory paths."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    cases = manifest['cases'] if isinstance(manifest, dict) else manifest
    if not isinstance(cases, list) or not cases:
        raise ValueError('Nonempty EpisodeSpec case list required')
    result, seen = [], set()
    for index, case in enumerate(cases):
        identity = str(case.get('case_id', f'case_{index:04d}'))
        if identity in seen or Path(identity).name != identity or identity in ('.', '..'):
            raise ValueError('Case IDs must be unique directory basenames')
        seen.add(identity)
        if 'episode_path' not in case:
            raise ValueError('Each case needs an explicit materialized episode_path; '
                             'a source HDF5 inventory alone is not executable')
        episode = Path(case['episode_path'])
        if not episode.is_absolute():
            episode = path.parent / episode
        blob = episode.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if case.get('episode_sha256') and digest != case['episode_sha256']:
            raise ValueError('EpisodeSpec changed: ' + identity)
        result.append(dict(case, case_id=identity, episode_path=str(episode.resolve()),
                           episode_sha256=digest))
    return result


def observed_points(observation):
    """Normalize native point arrays to one visible normalized point per camera.

    Native MolmoSpaces observations use normalized [0,1] image coordinates.
    Multiple handle projections are averaged, matching the Train extractor. No
    fallback projection from an expert pose or simulator handle is permitted.
    """
    result = {}
    for camera in CAMERAS:
        values = observation.get('object_image_points', {}).get('door_handle', {}).get(camera, [])
        count = None
        if isinstance(values, dict):
            count = int(np.asarray(values.get('num_points', [0])).reshape(-1)[0])
            values = values.get('points', [])
        array = np.asarray(values, dtype=float)
        if array.size:
            array = array.reshape(-1, 2)
            if count is not None:
                if not 0 <= count <= len(array):
                    raise ValueError('Native visible point count is outside padded point array')
                array = array[:count]
            if len(array):
                valid = np.isfinite(array).all(axis=1) & (array >= 0).all(axis=1) & (array <= 1).all(axis=1)
                if not valid.all():
                    raise ValueError('Invalid visible normalized door-handle point')
                average = array.mean(axis=0)
                result[camera] = (float(average[0]), float(average[1]), True)
                continue
        result[camera] = (0., 0., False)
    return result


def predicted_world_target(prediction, robot_base_world):
    from grasp.runtime import pose_matrix
    if prediction['arm'] not in ('left', 'right'):
        raise ValueError('Predicted arm must be left or right')
    base = np.asarray(robot_base_world, dtype=float)
    if base.shape != (4, 4) or not np.isfinite(base).all():
        raise ValueError('Finite base-to-world transform required')
    pose = np.asarray(prediction['tcp_base_xyz_wxyz'], dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1.) > .02:
        raise ValueError('Finite xyz+wxyz TCP prediction with unit quaternion required')
    return base @ pose_matrix(pose)


def oracle_prediction(reference, robot_base_world):
    """Express the exact supervised expert world TCP in the current base frame.

    This is a fixed source grasp pose, not a future command replay or a
    force-certified optimal grasp. Target updates obey the learned method's
    same interval and gate; no extra handle tracking is introduced.
    """
    from grasp.runtime import pose_matrix
    from scipy.spatial.transform import Rotation
    if reference.get('source') != 'expert_recorded_initial_grasp_tcp_world':
        raise ValueError('Explicit expert-reference oracle provenance required')
    arm = reference.get('arm')
    pose = np.asarray(reference.get('tcp_world_xyz_wxyz'), dtype=float)
    base = np.asarray(robot_base_world, dtype=float)
    if arm not in ('left', 'right') or pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Invalid oracle arm or expert world pose')
    if abs(np.linalg.norm(pose[3:])-1.) > .02 or base.shape != (4,4) or not np.isfinite(base).all():
        raise ValueError('Invalid oracle reference transform')
    world = pose_matrix(pose)
    relative = np.linalg.solve(base, world)
    quaternion = Rotation.from_matrix(relative[:3,:3]).as_quat(scalar_first=True)
    return dict(tcp_base_xyz_wxyz=np.r_[relative[:3,3],quaternion].tolist(),arm=arm,
        gripper_width_m=reference['gripper_width_m'],
        target_source='expert_recorded_initial_grasp_tcp_world',
        source_target_frame=reference['source_target_frame'])


def correction_action(task, arm, target, proposal, contact_good):
    from grasp.native_adapter import (AbsoluteJointTargets, NativeDeltaLimits,
                                      absolute_to_native_action, hold_native_action)
    from grasp.runtime import pose_error
    from grasp.servo import servo_delta
    view = task.env.robots[0].robot_view
    qpos = {name: value.copy() for name, value in view.get_qpos_dict().items()}
    actual = view.get_move_group(arm + '_gripper').leaf_frame_to_world.copy()
    error = pose_error(actual, target)
    action = hold_native_action(qpos, proposal, close_arm=arm)
    # Verified contact keeps the gripper closed; otherwise align before closing.
    close = contact_good or (error['position_m'] <= .020 and error['orientation_deg'] <= 5.)
    action[arm + '_gripper'] = np.array([100. if close else -100.])
    jacobian = view.get_jacobian(arm + '_gripper', [arm + '_arm'])
    cfg = dict(gain=.7, damping=.001, arm_step_rad=.04, base_step_m=.015)
    delta = servo_delta(jacobian, actual, target, False, cfg)
    limits = view.get_move_group(arm + '_arm').joint_pos_limits
    absolute = np.clip(qpos[arm + '_arm'] + delta, limits[:, 0], limits[:, 1])
    action = absolute_to_native_action(AbsoluteJointTargets({arm + '_arm': absolute}),
                                      qpos, policy_action=action, torso_mode='hold',
                                      limits=NativeDeltaLimits())
    return action, error


def run_case(args, case):
    import imageio.v2 as imageio
    import torch
    from grasp.contacts import ContactProbe
    from grasp.metrics import FINAL_POLICY, force_grasp, retained_grasp_opening
    from grasp.native_adapter import (GateConfig, LocalGraspGate, account_local_step,
                                      invalidate_action_buffer)
    from grasp.runtime import state, physics_digest, reset_state, restore_point_arrays
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    from olmo.eval.configure_molmo_spaces import MolmoBotRBY1DoorPlusOpenEvalConfig
    from pose_learning.predict import PosePredictor

    spec = EpisodeSpec.model_validate_json(Path(case['episode_path']).read_text())
    predictor = None
    if 'predicted_pose' in args.methods:
        predictor = PosePredictor.from_checkpoint(args.pose_checkpoint, device=args.device,
                                                  encoder_path=args.encoder)
    out = args.output / case['case_id']
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'case.json', case)
    save(out / 'status.json', dict(state='running', methods=args.methods))
    def identity(path):
        path = Path(path)
        return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns)
    receipt = dict(seed=args.seed, horizon=args.horizon, prediction_interval=args.prediction_interval,
                   pose_checkpoint=identity(args.pose_checkpoint) if args.pose_checkpoint else None,
                   molmobot_weight=identity(args.molmobot_checkpoint / 'model.pt'),
                   code_sha256={str(path.relative_to(Path(__file__).parents[1])):
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for package in ('pose_learning', 'grasp')
                       for path in (Path(__file__).parents[1] / package).glob('*.py')},
                   encoder_identity='Bound to the Train checkpoint and checked by PosePredictor',
                   oracle_pose_or_expert_action_input='oracle_pose' in args.methods,
                   oracle_pose_input_methods=['oracle_pose'] if 'oracle_pose' in args.methods else [],
                   expert_action_input=False,
                   contact_feedback_used_by_local_controller=True,
                   initial_image_tolerance_uint8=2)
    if args.pose_checkpoint:
        receipt['pose_checkpoint']['sha256'] = hashlib.sha256(args.pose_checkpoint.read_bytes()).hexdigest()
    save(out / 'receipt.json', receipt)
    reference = None
    policy = None
    results = {}
    for method in args.methods:
        directory = out / method
        directory.mkdir()
        sampler = writer = None
        try:
            os.environ['RBY1_MODEL_ACTION_SEED'] = str(args.seed)
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            config = MolmoBotRBY1DoorPlusOpenEvalConfig()
            config.policy_config.checkpoint_path = str(args.molmobot_checkpoint)
            config.task_horizon = args.horizon
            config.terminate_upon_success = False
            config.num_workers = 1; config.use_wandb = False; config.output_dir = directory
            config.scene_dataset, config.data_split = spec.scene_dataset, spec.data_split
            noise = getattr(config.robot_config, 'action_noise_config', None)
            save(directory / 'runtime.json', dict(
                action_noise_config=noise.model_dump(mode='json') if hasattr(noise, 'model_dump') else repr(noise),
                initial_state_source='materialized EpisodeSpec', seed=args.seed,
                task_horizon=args.horizon, metric_policy=FINAL_POLICY))
            config.task_sampler_config.house_inds = [spec.house_index]
            for name in ('randomize_lighting', 'randomize_textures', 'randomize_dynamics'):
                setattr(config.task_sampler_config, name, False)
            sampler = JsonEvalTaskSampler(config, spec)
            task = sampler.sample_task(house_index=spec.house_index)
            if task is None:
                raise RuntimeError('EpisodeSpec could not be loaded')
            if policy is None:
                policy = config.policy_config.policy_cls(config, task)
            policy.config = config
            policy.action_generator_seed = args.seed
            task.register_policy(policy)
            model, data = task.env.mj_model, task.env.mj_datas[0]
            if reference is not None:
                if physics_digest(model) != reference['physics']:
                    raise RuntimeError('Physics differ between paired methods')
                reset_state(model, data, reference['integration_state'])
            # Policy construction can consume RNG state; reseed each paired reset.
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            observation, _ = task.reset()
            obs = observation[0]
            if reference is not None:
                for name, value in reference['qpos'].items():
                    np.testing.assert_allclose(obs['qpos'][name], value, atol=1e-7, rtol=0)
                for name, value in reference['images'].items():
                    if np.abs(obs[name].astype(float) - value.astype(float)).max() > 2:
                        raise RuntimeError('Paired initial camera image differs by more than 2/255')
                    # Use the same audited conditioning pixels on both methods.
                    obs[name] = value.copy()
                obs['object_image_points'] = restore_point_arrays(copy.deepcopy(reference['points']))
            initial = dict(physics=physics_digest(model),
                           state=hashlib.sha256(state(model, data).tobytes()).hexdigest(),
                           input=fingerprint({k: obs[k] for k in (*CAMERAS, 'qpos', 'object_image_points') if k in obs}))
            if reference is None:
                reference = dict(initial, integration_state=state(model, data),
                                 images={name: obs[name].copy() for name in CAMERAS},
                                 qpos={name: value.copy() for name, value in obs['qpos'].items()},
                                 points=copy.deepcopy(obs.get('object_image_points', {})))
                np.save(directory.parent / 'initial_integration_state.npy', reference['integration_state'])
                save(directory.parent / 'initial_identity.json', initial)
            elif initial != {key: reference[key] for key in initial}:
                raise RuntimeError('Matched initial-state/input audit failed; no comparable score emitted')
            probes = {arm: ContactProbe(task, arm) for arm in ('left', 'right')}
            gate = LocalGraspGate(GateConfig(max_local_steps=40, max_events=2, cooldown_steps=16))
            traces = {arm: [] for arm in probes}
            target = None; prediction = None; arm = None; proposal = None
            first_action = None; interventions = predictions = 0
            condition = None
            # A frame-zero measurement is needed to detect opening before grasp.
            for side, probe in probes.items():
                actual = task.env.robots[0].robot_view.get_move_group(side + '_gripper').leaf_frame_to_world
                sample = probe.measure(task, side, actual)
                sample.update(step=0, stage=method); traces[side].append(sample)
            if args.video:
                writer = imageio.get_writer(directory / 'head_camera.mp4', fps=10, macro_block_size=1)
                writer.append_data(obs['head_camera'])
            began = time.perf_counter()
            with torch.inference_mode(), (directory / 'events.jsonl').open('w') as stream:
                for step in range(args.horizon):
                    obs = observation[0]
                    view = task.env.robots[0].robot_view
                    force = {side: force_grasp(trace[-1], FINAL_POLICY)[0]
                             for side, trace in traces.items()}
                    if method != 'baseline' and not gate.active and step % args.prediction_interval == 0:
                        if method == 'predicted_pose':
                            prediction = predictor.predict(
                                images={name: obs[name] for name in CAMERAS}, qpos=obs['qpos'],
                                points=observed_points(obs),
                                task_is_pull='pull' in spec.language.task_description.casefold())
                        else:
                            prediction = oracle_prediction(case['oracle_pose'],task.env.robots[0].get_world_pose_tf_mat())
                        target = predicted_world_target(prediction, task.env.robots[0].get_world_pose_tf_mat())
                        arm = prediction['arm']; predictions += 1
                    queried = False
                    if method == 'baseline' or not gate.active:
                        proposal = copy.deepcopy(policy.get_action(observation)); queried = True
                    if step == 0:
                        first_action = fingerprint(proposal)
                        condition = fingerprint(dict(image=policy._conditioning_image, points=policy._conditioning_points))
                        if method == args.methods[0]:
                            reference['first_action'] = first_action; reference['conditioning'] = condition
                        elif first_action != reference['first_action'] or condition != reference['conditioning']:
                            raise RuntimeError('Initial policy proposal/conditioning differs between methods')
                    details = {}; decision = None
                    if method == 'baseline':
                        action = proposal
                    else:
                        distance = {arm: float(np.linalg.norm(view.get_move_group(arm + '_gripper').leaf_frame_to_world[:3, 3] - target[:3, 3]))}
                        decision = gate.decide(proposal if queried else None, distance, force)
                        if decision.finished:
                            invalidate_action_buffer(policy)
                            proposal = copy.deepcopy(policy.get_action(observation)); queried = True
                        if decision.use_local:
                            if decision.started:
                                invalidate_action_buffer(policy)
                            action, details = correction_action(task, decision.arm, target, proposal, force[decision.arm])
                            account_local_step(policy, observation, model_was_queried=queried)
                            interventions += 1
                        else:
                            action = proposal
                    if action is None or not all(np.isfinite(v).all() for v in action.values()):
                        raise ValueError('Invalid native action')
                    np.random.seed(args.seed + 100000 + step)
                    observation, _, terminal, truncated, _ = task.step(copy.deepcopy(action))
                    if policy.step_count != step + 1 or len(policy.obs_history) != step + 1:
                        raise RuntimeError('Policy history does not match executed controls')
                    if fingerprint(dict(image=policy._conditioning_image, points=policy._conditioning_points)) != condition:
                        raise RuntimeError('Initial policy conditioning changed')
                    for side, probe in probes.items():
                        actual = view.get_move_group(side + '_gripper').leaf_frame_to_world
                        sample = probe.measure(task, side, actual)
                        sample.update(step=step + 1, stage=method); traces[side].append(sample)
                    strict = {side: retained_grasp_opening(trace, task.config.task_config.door_openness_threshold,
                                                          policy=FINAL_POLICY)
                              for side, trace in traces.items()}
                    stream.write(json.dumps(dict(step=step + 1, local=bool(decision and decision.use_local),
                                                 predicted_pose=prediction if method=='predicted_pose' else None,
                                                 oracle_pose=prediction if method=='oracle_pose' else None,
                                                 alignment_error=details,
                                                 action=action), default=plain) + '\n')
                    if writer:
                        writer.append_data(observation[0]['head_camera'])
                    if any(x['retained_grasp_opening_success'] for x in strict.values()) or np.any(terminal) or np.any(truncated):
                        break
            threshold = task.config.task_config.door_openness_threshold
            result = dict(state='complete', method=method, steps=step + 1,
                          evaluation_scope='qualified_model_test' if case.get('model_evaluation_ready') is True
                                           else 'diagnostic_case_not_certified_for_final_model_test',
                          case_retained_grasp_verified=case.get('stable_grasp_feasibility_verified', False),
                          door_opening_success=any(x['opening_fraction'] >= threshold for x in traces['left']),
                          retained_grasp_opening_success=any(x['retained_grasp_opening_success'] for x in strict.values()),
                          per_arm_retention=strict, metric_policy=FINAL_POLICY,
                          intervention_steps=interventions, pose_queries=predictions,
                          pose_predictions=predictions if method=='predicted_pose' else 0,
                          oracle_pose_queries=predictions if method=='oracle_pose' else 0,
                          wall_seconds=time.perf_counter() - began, matched_inputs=True,
                          target_source={'baseline':None,'predicted_pose':'predicted_observation_base_tcp',
                                         'oracle_pose':'expert_reference_fixed_world_tcp'}[method],
                          oracle_pose_input=method=='oracle_pose', expert_action_input=False,
                          learned_gripper_width_used=False,
                          gripper_execution='Native open/close guarded by pose error and bilateral contact',
                          seed=args.seed, horizon=args.horizon)
            for side, trace in traces.items():
                save(directory / f'{side}_trace.json', trace)
            save(directory / 'summary.json', result); results[method] = result
        except BaseException as error:
            save(directory / 'status.json', dict(state='failed_or_incomparable', error=repr(error)))
            save(out / 'status.json', dict(state='failed_or_incomparable', method=method,
                                           error=repr(error), completed_methods=list(results)))
            raise
        finally:
            if writer is not None:
                writer.close()
            if sampler is not None:
                sampler.close()
    save(out / 'summary.json', dict(case_id=case['case_id'], results=results, state='complete'))
    save(out / 'status.json', dict(state='complete', methods=args.methods, paired=True))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--methods', nargs='+', choices=('baseline', 'predicted_pose','oracle_pose'), default=['baseline', 'predicted_pose'])
    parser.add_argument('--molmobot-checkpoint', type=Path, default=Path('/bigtemp/xna8aw/LRL/hf_models/MolmoBot-RBY1Multitask'))
    parser.add_argument('--pose-checkpoint', type=Path)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--prediction-interval', type=int, default=8)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.horizon < 1 or args.prediction_interval < 1 or len(set(args.methods)) != len(args.methods):
        parser.error('Positive control budgets and unique methods required')
    if 'predicted_pose' in args.methods and args.pose_checkpoint is None:
        parser.error('--pose-checkpoint is required for predicted_pose')
    cases = load_cases(args.episodes)
    if 'oracle_pose' in args.methods:
        if any('oracle_pose' not in case for case in cases):
            parser.error('oracle_pose requires a reference for every requested case; no silent baseline fallback')
        for case in cases:
            oracle_prediction(case['oracle_pose'],np.eye(4))
    if args.index is not None:
        if not 0 <= args.index < len(cases):
            parser.error('--index outside case manifest')
        cases = [cases[args.index]]
    from grasp.runtime import select_allocated_egl_device
    select_allocated_egl_device()
    import torch
    torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    args.output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        run_case(args, case)


if __name__ == '__main__':
    main()
