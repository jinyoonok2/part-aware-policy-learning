"""Construct a measured expert witness from an exact benchmark EpisodeSpec.

The benchmark start is authoritative. This worker never samples a new task,
replays a learned policy, or teleports the robot along a reference trajectory.
An unsuccessful expert attempt is preserved and does not prove impossibility.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback


def plain(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, type): return value.__module__ + '.' + value.__qualname__
    return str(value)


def configuration_for_json(value):
    """Record unbounded configuration values without changing live settings.

    JSON has no infinity literal. Only configuration snapshots use the explicit
    strings below; measurements and metrics must still pass strict finite JSON.
    """
    import numpy as np
    if isinstance(value, np.ndarray):
        return configuration_for_json(value.tolist())
    if isinstance(value, np.generic):
        return configuration_for_json(value.item())
    if isinstance(value, dict):
        return {key: configuration_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN is not a valid expert configuration value')
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    default=plain, allow_nan=False).encode()).hexdigest()


def inside(root, relative):
    root = Path(root).resolve(); path = root / relative
    if not path.resolve().is_relative_to(root): raise ValueError('Artifact path escapes its run')
    return path


def verify_inputs(run, study, row):
    for name, expected in study.get('source_hashes', {}).items():
        if digest(inside(run/'source', name)) != expected:
            raise ValueError('Frozen worker source changed: ' + name)
    path = inside(run, row['episode_path'])
    if digest(path) != row['episode_sha256']: raise ValueError('Frozen benchmark episode changed')
    train_path = study.get('train_index_path', study.get('train_index', study.get('train_inventory')))
    if train_path:
        expected = study.get('train_index_sha256', study.get('train_inventory_sha256'))
        if expected and digest(train_path) != expected: raise ValueError('Train inventory changed')
        rows = [json.loads(line) for line in Path(train_path).read_text().splitlines() if line.strip()]
        if row['house'] in {item['house'] for item in rows}:
            raise ValueError('Benchmark Test house overlaps prepared Train')
    return path


def verify_backend(study):
    for package, field in [('torch', 'torch_version'), ('mujoco', 'mujoco_version')]:
        if study.get(field) and importlib.metadata.version(package) != study[field]:
            raise ValueError('Pinned package changed: ' + package)
    dist = importlib.metadata.distribution('nvidia-curobo')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    actual = direct.get('vcs_info', {}).get('commit_id')
    receipt = Path(study.get('curobo_build_receipt', os.environ.get('PART_CUROBO_BUILD_RECEIPT', '/nonexistent')))
    if actual is None and receipt.is_file():
        build = json.loads(receipt.read_text())
        actual = build.get('curobo_revision')
        if build.get('installed_record_sha256') != hashlib.sha256((dist.read_text('RECORD') or '').encode()).hexdigest():
            raise ValueError('Planner receipt does not bind the installed wheel')
    if actual != study['curobo_revision']: raise ValueError('Wrong or unverifiable native planner revision')


def initial_alignment(spec, robot_qpos, base_pose, door_position, object_poses,
                      qpos_tolerance=1e-5, position_tolerance=.005, rotation_tolerance_deg=1.):
    """Compare LIVE initial values to the spec, never source to itself."""
    import numpy as np
    from grasp.runtime import pose_error, pose_matrix
    joint_errors = {}
    for name, expected in spec['robot']['init_qpos'].items():
        actual = np.asarray(robot_qpos[name]); expected = np.asarray(expected)
        if actual.shape != expected.shape: raise ValueError('Initial robot shape mismatch: ' + name)
        joint_errors[name] = float(np.max(np.abs(actual - expected)))
    base_error = pose_error(np.asarray(base_pose), pose_matrix(spec['task']['robot_base_pose']))
    door_error = float(np.max(np.abs(np.asarray(door_position) -
                                     np.asarray(spec['task']['articulated_joint_reset_state']))))
    objects = {}
    for name, expected in spec.get('scene_modifications', {}).get('object_poses', {}).items():
        if name not in object_poses: raise ValueError('Recorded object absent from live model: ' + name)
        objects[name] = pose_error(np.asarray(object_poses[name]), pose_matrix(expected))
    passed = (max(joint_errors.values(), default=0.) <= qpos_tolerance
              and door_error <= qpos_tolerance
              and base_error['position_m'] <= position_tolerance
              and base_error['orientation_deg'] <= rotation_tolerance_deg
              and all(x['position_m'] <= position_tolerance and x['orientation_deg'] <= rotation_tolerance_deg
                      for x in objects.values()))
    return dict(passed=bool(passed), robot_joint_max_errors=joint_errors, base_pose_error=base_error,
                door_joint_error=door_error, object_pose_errors=objects,
                tolerances=dict(joint=qpos_tolerance, position_m=position_tolerance,
                                orientation_deg=rotation_tolerance_deg))


def continuation_count(executed, horizon, requested):
    if not (0 <= executed <= horizon and 0 <= requested <= 20):
        raise ValueError('Invalid action budget or continuation request')
    return min(requested, horizon - executed)


def request_expert_action(expert, observation):
    """Treat the pinned planner's retry exhaustion as an expert termination.

    The completed state/action prefix is still a valid recorded attempt. Other
    exceptions remain technical errors and must not be hidden as task failures.
    """
    try:
        return expert.get_action(observation), None
    except ValueError as error:
        if str(error) != 'Max planning reattempts reached.':
            raise
        return None, dict(reason='planner_retry_limit', exception_type='ValueError',
                          message=str(error))


def verify_assets(run, study, row, index):
    scope = row.get('asset_stage_scope', 'smoke' if index == study.get('smoke_case_index') else 'all')
    if scope not in ('smoke', 'all') and not (scope.startswith('batch_') and scope[6:].isdigit()):
        raise ValueError('Invalid bounded asset-stage scope')
    staged = json.loads((run/f'staged_{scope}.json').read_text())
    if staged.get('state') != 'passed' or staged.get('benchmark_sha256') != study['benchmark_sha256']:
        raise ValueError('Exact benchmark assets have not passed staging')
    if digest(run/f'assets_{scope}/manifest.json') != staged['asset_manifest_sha256']:
        raise ValueError('Asset staging manifest changed')
    membership = staged.get('case_episode_sha256')
    if scope.startswith('batch_') and (not isinstance(membership, dict) or membership.get(row['case_id']) != row['episode_sha256']):
        raise ValueError('Case is outside the staged verification batch')
    return staged


def checked_existing(case, episode_sha256):
    path = case/'summary.json'
    if not path.exists(): return None
    summary = json.loads(path.read_text())
    if summary.get('state') != 'complete': return None
    if summary.get('episode_sha256') != episode_sha256: raise ValueError('Completed episode identity changed')
    for relative, expected in summary['artifact_sha256'].items():
        if digest(inside(case, relative)) != expected: raise ValueError('Completed expert artifact changed')
    return summary


def worker(args):
    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from grasp.metrics import FINAL_POLICY, retained_grasp_opening
    from grasp.runtime import select_allocated_egl_device, integration_state as state
    from grasp.runtime import physics_digest
    from grasp.contacts import ContactProbe
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    from molmo_spaces.data_generation.config.door_opening_configs import DoorOpeningDataGenConfig

    run=args.run;study=json.loads((run/'study.json').read_text());row=study['cases'][args.index]
    if study.get('worker_source_frozen') is not True:
        raise RuntimeError('Exact-start worker sources have not been finalized')
    case=run/'cases'/row['case_id'];case.mkdir(parents=True,exist_ok=True)
    episode_path=verify_inputs(run,study,row)
    verify_assets(run,study,row,args.index)
    if checked_existing(case,row['episode_sha256']):
        print('Completed expert witness already verified: '+row['case_id'],flush=True);return
    if (case/'worker_started.json').exists():
        raise RuntimeError('Preserve prior incomplete attempt; archive it explicitly before retrying this case')
    write(case/'worker_started.json',dict(pid=os.getpid(),index=args.index,episode_sha256=row['episode_sha256']))
    verify_backend(study)
    select_allocated_egl_device();torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('Native expert requires an allocated CUDA device')
    seed=int(study.get('audit_seed',42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    spec=EpisodeSpec.model_validate_json(episode_path.read_text())
    if spec.data_split!='val' or spec.house_index!=row['house']: raise ValueError('Wrong benchmark source identity')
    policy=copy.deepcopy(study.get('metric_policy',FINAL_POLICY))
    if canonical_hash(policy)!=canonical_hash(FINAL_POLICY): raise ValueError('Frozen metric and imported implementation differ')
    horizon=int(study.get('horizon',400));timing=study.get('timing_ms',dict(policy=100.,control=20.,simulation=4.))
    began=time.perf_counter();sampler=None;writers={};task=None;states=[];qpos=[];qvel=[];controls=[]
    traces={'left':[],'right':[]};commands=[];substeps=[];robot_states=[];phase_values=[];stages=[];successes=[]
    frames=[];extended=study.get('witness_format')=='benchmark_expert_witness_v2'
    exact=False;alignment=None;initial_eligible=False;prefix_steps=0;hold_steps=0;prefix_metric={}
    try:
        write(case/'status.json',dict(state='loading_exact_benchmark_scene',case_id=row['case_id'],index=args.index))
        cfg=DoorOpeningDataGenConfig(scene_dataset=spec.scene_dataset,data_split='val',seed=seed,
            output_dir=case,num_workers=1,use_passive_viewer=False,use_wandb=False,
            task_horizon=horizon,terminate_upon_success=False,end_on_success=False,
            policy_dt_ms=timing['policy'],ctrl_dt_ms=timing['control'],sim_dt_ms=timing['simulation'],
            filter_for_successful_trajectories=False)
        cfg.robot_config.action_noise_config.enabled=False
        cfg.task_sampler_config.house_inds=[spec.house_index]
        for name in ('randomize_lighting','randomize_textures','randomize_dynamics'):
            if hasattr(cfg.task_sampler_config,name):setattr(cfg.task_sampler_config,name,False)
        if cfg.policy_config is None:raise RuntimeError('Native planner configuration was not initialized')
        for planner in (cfg.policy_config.left_curobo_planner_config,cfg.policy_config.right_curobo_planner_config):
            planner.collision_cache={'mesh':500,'obb':80}
        # The native policy class must already be configured here so its
        # collision-check helper bodies are added before compiling the scene.
        sampler=JsonEvalTaskSampler(cfg,spec)
        task=sampler.sample_task(house_index=spec.house_index)
        if task is None:raise RuntimeError('JSON sampler did not construct the benchmark task')
        robot=task.env.robots[0];view=robot.robot_view;model,data=task.env.mj_model,task.env.mj_datas[0]
        expert=cfg.policy_config.policy_cls(cfg,task);task.register_policy(expert)
        observation,_=task.reset()
        live_qpos={name:view.get_move_group(name).joint_pos.copy() for name in spec.robot.init_qpos}
        object_poses={}
        for name in spec.scene_modifications.object_poses:
            body=model.body(name).id;matrix=np.eye(4);matrix[:3,3]=data.xpos[body]
            matrix[:3,:3]=data.xmat[body].reshape(3,3);object_poses[name]=matrix
        alignment=initial_alignment(spec.model_dump(),live_qpos,view.base.pose.copy(),
            task.door_object.get_joint_position(task.door_object.get_hinge_joint_index()),object_poses)
        write(case/'initial_alignment.json',alignment)
        if not alignment['passed']:raise RuntimeError('Exact benchmark initial-state alignment failed')
        exact=True
        write(case/'resolved_config.json',dict(native_config=configuration_for_json(cfg.model_dump()),episode_spec=spec.model_dump(),
            native_config_infinity_encoding='Infinity and -Infinity strings represent unbounded configuration values; live settings are unchanged',
            expert_policy=type(expert).__module__+'.'+type(expert).__name__,source='new_privileged_expert_from_exact_benchmark_start',
            original_historical_expert_replay=False,noise_enabled=False,full_controller_equivalence_established=False,
            compatibility_note='Absolute expert joint controllers and runtime collision helper bodies are declared. Future model replay/control equivalence must be checked before final evaluation.'))
        np.savez_compressed(case/'initial_state.npz',integration_state=state(model,data),qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),actuator_control=data.ctrl.copy(),base_world=view.base.pose.copy())
        # MjSpec.compile() does not populate MuJoCo's global last-XML buffer.
        mujoco.mj_saveModel(model,str(case/'compiled_initial_scene.mjb'),None)
        probes={arm:ContactProbe(task,arm) for arm in traces}
        camera_names=[name for name in ('head_camera','wrist_camera_l','wrist_camera_r')
                      if name in [c.name for c in cfg.camera_config.cameras]]
        for camera in camera_names:
            writers[camera]=imageio.get_writer(case/(camera+'.mp4'),fps=1000./timing['policy'],macro_block_size=1)
        if 'head_camera' not in writers:raise RuntimeError('Recorded benchmark has no head camera')
        def sample(step,stage):
            for arm,probe in probes.items():
                entry=probe.measure(task,arm,np.eye(4));entry.pop('tcp_error',None)
                entry.update(step=step,stage=stage,robot_base_world=view.base.pose.copy());traces[arm].append(entry)
            states.append(state(model,data));qpos.append(data.qpos.copy());qvel.append(data.qvel.copy());controls.append(data.ctrl.copy())
            robot_states.append({name:value.copy() for name,value in view.get_qpos_dict().items()})
            phase_values.append(int(getattr(expert,'phase_value',-1)));stages.append(stage);successes.append(bool(task.judge_success()))
            if extended:
                from data_preparation.witness_observations import capture
                frames.append(capture(task,observation,camera_names))
            obs=observation[0] if isinstance(observation,(list,tuple)) else observation
            for camera,writer in writers.items():
                writer.append_data(obs[camera] if extended else task.env.render_rgb_frame(camera))
        sample(0,'initial')
        initial_opening=traces['left'][0]['opening_fraction']
        initial_eligible=initial_opening<=policy['max_opening_at_grasp'] and not bool(task.judge_success())
        original_step=task.env.step
        def recorded_step(n_steps=1):
            substeps.append(dict(policy_step=int(task.episode_step_count),n_sim_steps=int(n_steps),
                simulation_time=float(data.time),actuator_controls=data.ctrl.copy()))
            return original_step(n_steps)
        task.env.step=recorded_step
        last_action=None;stop_reason='initial_state_ineligible';planner_termination=None
        if initial_eligible:
            stop_reason='action_budget'
            while int(task.episode_step_count)<horizon:
                requested,planner_termination=request_expert_action(expert,observation)
                if planner_termination is not None:
                    stop_reason=planner_termination['reason'];break
                if requested is None:stop_reason='expert_returned_none';break
                if requested.get('done',False):stop_reason='expert_done_before_opening';break
                last_action=copy.deepcopy(requested)
                observation,_,_,_,_=task.step(copy.deepcopy(requested))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise FloatingPointError('Nonfinite expert physics')
                step=int(task.episode_step_count)
                commands.append(dict(step=step,stage='expert_prefix',action=last_action,
                    applied_controller_targets={name:c.target_pos.copy() for name,c in robot.controllers.items() if hasattr(c,'target_pos')}))
                sample(step,'expert_prefix')
                if successes[-1]:stop_reason='first_ordinary_opening';break
                if bool(np.all(task.is_done())):stop_reason='native_terminal_or_budget';break
                if step%25==0:write(case/'status.json',dict(state='executing_exact_start_expert',case_id=row['case_id'],step=step))
            prefix_steps=int(task.episode_step_count)
            threshold=float(task.config.task_config.door_openness_threshold)
            prefix_metric={arm:retained_grasp_opening(trace,threshold,policy) for arm,trace in traces.items()}
            if successes[-1] and last_action is not None:
                for _ in range(continuation_count(prefix_steps,horizon,int(study.get('continuation_steps',10)))):
                    observation,_,_,_,_=task.step(copy.deepcopy(last_action));step=int(task.episode_step_count)
                    if step==prefix_steps+hold_steps:raise RuntimeError('Native task refused declared continuation physics')
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise FloatingPointError('Nonfinite continuation physics')
                    hold_steps+=1
                    commands.append(dict(step=step,stage='last_command_hold',action=copy.deepcopy(last_action),
                        applied_controller_targets={name:c.target_pos.copy() for name,c in robot.controllers.items() if hasattr(c,'target_pos')}))
                    sample(step,'last_command_hold')
        task.env.step=original_step
        for writer in writers.values():writer.close()
        writers={}
        threshold=float(task.config.task_config.door_openness_threshold)
        measured={arm:retained_grasp_opening(trace,threshold,policy) for arm,trace in traces.items()}
        passed=initial_eligible and any(value['retained_grasp_opening_success'] for value in measured.values())
        poses={}
        for arm,value in measured.items():
            if value['retained_grasp_opening_success']:
                onset=next(x for x in traces[arm] if x['step']==value['grasp_onset_step'])
                poses[arm]=dict(grasp_onset_step=onset['step'],tcp_world=onset['tcp_world'],
                    tcp_in_robot_base=np.linalg.inv(np.asarray(onset['robot_base_world']))@np.asarray(onset['tcp_world']),
                    tcp_in_handle=np.linalg.inv(np.asarray(onset['selected_handle_world']))@np.asarray(onset['tcp_world']),
                    gripper_joint_pos=onset['gripper_joint_pos'],origin='Measured newly generated expert witness')
        write(case/'grasp_poses.json',poses)
        for arm,trace in traces.items():write(case/(arm+'_force_trace.json'),trace)
        for name,rows in [('executed_commands.jsonl',commands),('applied_control_substeps.jsonl',substeps)]:
            with (case/name).open('w') as stream:
                for entry in rows:stream.write(json.dumps(entry,default=plain,allow_nan=False)+'\n')
        _save_h5(case,study,row,spec,states,qpos,qvel,controls,robot_states,traces,phase_values,stages,successes,commands,camera_names,
            frames=frames if extended else None,
            scene_metadata=dict(task_description=spec.language.task_description,policy_dt_ms=timing['policy'],
                policy_phases=expert.get_all_phases(),origin='new exact-start native expert',
                episode_sha256=row['episode_sha256'],native_config_reference='resolved_config.json'))
        if extended:
            from data_preparation.export_benchmark_pose_labels import export_case
            export_case(run,row)
        filenames=['initial_state.npz','expert_trajectory.h5','left_force_trace.json','right_force_trace.json',
            'executed_commands.jsonl','applied_control_substeps.jsonl','head_camera.mp4','grasp_poses.json',
            'initial_alignment.json','resolved_config.json','compiled_initial_scene.mjb']
        filenames += [name+'.mp4' for name in camera_names if name!='head_camera']
        if extended:filenames+=['pose_label_status.json']
        if extended and (case/'pose_target.json').exists():filenames+=['pose_target.json']
        summary=dict(state='complete',case_id=row['case_id'],original_episode_index=row['original_episode_index'],
            house=row['house'],source_split='val',project_role='test',episode_sha256=row['episode_sha256'],
            origin='new_native_expert_witness_from_exact_benchmark_start',expert_policy=type(expert).__name__,
            exact_start_verified=exact,initial_alignment_passed=alignment['passed'],initial_eligible=initial_eligible,
            initial_opening_fraction=initial_opening,horizon=horizon,executed_actions=len(commands),
            prefix_steps=prefix_steps,continuation_steps=hold_steps,continuation_protocol='Repeat unchanged last expert command after first ordinary opening, within the same action budget; separate newly executed hold evidence.',
            stop_reason=stop_reason,planner_termination=planner_termination,
            recording_runtime_revision=study.get('runtime_revision'),worker_source_sha256=digest(__file__),
            ordinary_opening_ever=bool(any(successes)),threshold=threshold,
            metric_version=policy['metric_version'],metric_policy_sha256=canonical_hash(policy),
            source_prefix_retention=prefix_metric,retained_grasp_by_arm=measured,
            retained_grasp_witness_passed=bool(passed),qualified_for_model_test=False,final_model_test_ready=False,
            feasibility_status='measured_expert_witness' if passed else 'not_demonstrated_by_this_expert',
            intrinsic_task_infeasible=False,full_controller_equivalence_established=False,
            model_inference_used=False,training_started=False,baseline_outcomes_used=False,
            physics_sha256=physics_digest(model),elapsed_seconds=time.perf_counter()-began,
            reference_format=study.get('witness_format','benchmark_expert_witness_v1'),
            compiled_scene_artifact=dict(path='compiled_initial_scene.mjb',format='MuJoCo binary model'),
            artifact_sha256={name:digest(case/name) for name in filenames})
        assert summary['executed_actions']<=horizon
        write(case/'summary.json',summary);write(case/'status.json',summary)
        print(json.dumps({key:summary[key] for key in ('case_id','state','executed_actions','retained_grasp_witness_passed')},indent=2),flush=True)
    except Exception:
        # Runtime/compatibility failures remain unknown, but retain every
        # physical observation collected before the error for diagnosis.
        for arm,trace in traces.items():
            if trace:write(case/(arm+'_partial_force_trace.json'),trace)
        if states:
            np.savez_compressed(case/'partial_trajectory.npz',integration_states=np.asarray(states),
                qpos=np.asarray(qpos),qvel=np.asarray(qvel),actuator_controls=np.asarray(controls))
        for name,rows in [('partial_executed_commands.jsonl',commands),('partial_control_substeps.jsonl',substeps)]:
            if rows:
                with (case/name).open('w') as stream:
                    for entry in rows:stream.write(json.dumps(entry,default=plain,allow_nan=False)+'\n')
        raise
    finally:
        for writer in writers.values():writer.close()
        if sampler is not None:sampler.close()


def _save_h5(case,study,row,spec,states,qpos,qvel,controls,robot_states,traces,phases,stages,successes,commands,cameras,frames=None,scene_metadata=None):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    count=len(states)
    if (not count or any(len(value)!=count for value in [qpos,qvel,controls,robot_states,
            traces['left'],traces['right'],phases,stages,successes]) or len(commands)+1!=count):
        raise ValueError('Expert trajectory action/state/pose arrays are not aligned')
    def encoded(values):return np.asarray([json.dumps(value,default=plain,allow_nan=False) for value in values],dtype=h5py.string_dtype('utf-8'))
    def pose7(matrix):
        matrix=np.asarray(matrix);return np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat(scalar_first=True)]
    with h5py.File(case/'expert_trajectory.h5','w') as file:
        file.attrs['origin']='new exact-benchmark-start expert witness, not original released expert trajectory'
        file.attrs['episode_sha256']=row['episode_sha256'];file.attrs['format']=study.get('witness_format','benchmark_expert_witness_v1')
        group=file.create_group('traj_0')
        for key,values in [('integration_states',states),('simulator_qpos',qpos),('simulator_qvel',qvel),('actuator_controls',controls)]:
            group.create_dataset(key,data=np.asarray(values),compression='gzip')
        group.create_dataset('success',data=np.asarray(successes,dtype=bool))
        group.create_dataset('actions/commanded_action',data=encoded([{}]+[x['action'] for x in commands]))
        group.create_dataset('obs/agent/qpos',data=encoded(robot_states))
        group.create_dataset('obs/extra/policy_phase',data=np.asarray(phases,dtype=np.int32))
        group.create_dataset('obs/extra/execution_stage',data=np.asarray(stages,dtype=h5py.string_dtype('utf-8')))
        group.create_dataset('obs/extra/robot_base_pose',data=np.asarray([pose7(x['robot_base_world']) for x in traces['left']]))
        for arm,values in traces.items():
            group.create_dataset(f'obs/extra/{arm}_tcp_pose',data=np.asarray([pose7(np.linalg.inv(np.asarray(x['robot_base_world']))@np.asarray(x['tcp_world'])) for x in values]))
        group.create_dataset('obs/extra/task_info',data=encoded([dict(handle_grasped=left['benchmark_grasped']) for left in traces['left']]))
        group.create_dataset('obs/extra/door_state',data=encoded([dict(opening_percentage=x['opening_fraction'],handle_position=np.asarray(x['selected_handle_world'])[:3,3]) for x in traces['left']]))
        for camera in cameras:
            if frames is not None:
                group.create_dataset('obs/sensor_data/'+camera,data=np.frombuffer((camera+'.mp4').encode(),dtype=np.uint8))
            else:group.create_dataset('obs/sensor_data/'+camera,data=camera+'.mp4',dtype=h5py.string_dtype('utf-8'))
        if frames is not None:
            from data_preparation.witness_observations import save
            save(group,frames,cameras,encoded)
            group.create_dataset('obs_scene',data=json.dumps(scene_metadata,default=plain),dtype=h5py.string_dtype('utf-8'))
        group.create_dataset('episode_spec_json',data=json.dumps(spec.model_dump(),default=plain),dtype=h5py.string_dtype('utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['worker'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    try:worker(args)
    except Exception:
        study=json.loads((args.run/'study.json').read_text());row=study['cases'][args.index]
        case=args.run/'cases'/row['case_id'];case.mkdir(parents=True,exist_ok=True)
        value=dict(state='runtime_or_dependency_error',case_id=row['case_id'],error=traceback.format_exc(),
            feasibility_status='unknown',intrinsic_task_infeasible=False,retained_grasp_witness_passed=False,
            qualified_for_model_test=False,final_model_test_ready=False,training_started=False,model_inference_used=False)
        write(case/'worker_error.json',value);write(case/'status.json',value);raise
    finally:
        reporter=args.run/'source/report_benchmark_expert_verification.py'
        if reporter.is_file():
            result=subprocess.run(['/usr/bin/python3',str(reporter),'--run',str(args.run)],
                                  capture_output=True,text=True)
            if result.returncode:
                write(args.run/f'report_error_{os.getpid()}.json',dict(returncode=result.returncode,stderr=result.stderr[-4000:]))


if __name__=='__main__':main()
