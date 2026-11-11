"""Run the retained contact controller or guarded opening on frozen Train cases.

The contact default reproduces a local diagnostic after the expert approach.
Use --mode opening --start original for the full-start guarded opening test.
"""
import argparse
import json
import traceback
from pathlib import Path

import h5py
import numpy as np

from .contacts import ContactProbe
from .firm import run_guarded
from .metrics import FINAL_POLICY, POLICY, retained_grasp_opening
from .opening import joint_metrics, save_states
from .runtime import decode_row, digest, integration_state, physics_step, pose_matrix, select_allocated_egl_device, write
from .servo import BEST_PARAMETERS, REFERENCE_PARAMETERS, reconstruct_handle, run_contact

FIXTURE = Path(__file__).resolve().parent / 'fixtures/recovery_10/manifest.json'


def run_case(args):
    manifest = json.loads(args.manifest.read_text())
    row = manifest['cases'][args.index]
    dataset = args.dataset or Path(manifest['original_dataset'])
    source = dataset / row['h5']
    if digest(source) != row['h5_sha256']:
        raise ValueError('Source trajectory hash differs from the retained fixture')
    fixture = args.manifest.parent / f'{args.index:02d}'
    case = args.run / 'cases' / f'{args.index:02d}'
    case.mkdir(parents=True, exist_ok=False)
    try:
        write(case / 'status.json', dict(state='loading_scene'))
        select_allocated_egl_device()
        from olmo.eval.configure_molmo_spaces import RBY1RecordedTrajectoryEvalConfig
        from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
        from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
        with h5py.File(source, 'r') as h5:
            group = h5[row['trajectory']]
            source_door = decode_row(group['obs/extra/door_state'][row['interaction_event']['target_frame']])
            actions = [decode_row(value) for value in group['actions/commanded_action']]
        np.random.seed(42)
        cfg = RBY1RecordedTrajectoryEvalConfig()
        cfg.num_workers, cfg.use_wandb, cfg.task_horizon, cfg.output_dir = 1, False, 1000, case
        frozen = json.loads((fixture / 'decoded_config.json').read_text())['robot_config']
        for key in ('robot_xml_path', 'gravcomp', 'K_stiffness', 'K_damping'):
            if key in frozen:
                setattr(cfg.robot_config, key, Path(frozen[key]) if key == 'robot_xml_path' else frozen[key])
        spec = EpisodeSpec.model_validate_json((fixture / 'episode.json').read_text())
        original = JsonEvalTaskSampler(cfg, spec).sample_task(house_index=row['house'])
        if original is None:
            raise RuntimeError('Native task sampler did not produce the frozen case')
        prefix_steps = row['condition_frames'][-1]['frame'] if args.start == 'expert-pregrasp' else 0
        for frame in range(1, prefix_steps + 1):
            physics_step(original, actions[frame])
        initial = integration_state(original.env.mj_model, original.env.mj_datas[0])
        arm = row['interaction_event']['arm']
        probe = ContactProbe(original, arm)
        target = pose_matrix(row['interaction_event']['measured_tcp_world_xyz_wxyz'])
        audit = reconstruct_handle(original, probe, source_door, target)
        relative = np.asarray(audit['tcp_in_handle'])
        write(case / 'handle_frame_audit.json', audit)
        results = {}
        variants = [('combined', BEST_PARAMETERS)]
        if args.with_reference:
            if args.mode != 'contact':
                raise ValueError('--with-reference is only defined for the historical contact comparison')
            variants.insert(0, ('reference', REFERENCE_PARAMETERS))
        for name, parameters in variants:
            write(case / 'status.json', dict(state='running', controller=name, results=results))
            if args.mode == 'contact':
                trace, result = run_contact(original, arm, probe, relative, parameters)
            else:
                budget = args.max_steps - prefix_steps
                if budget <= 0:
                    raise ValueError('Expert approach exceeds the declared total action budget')
                trace, result, states = run_guarded(original, arm, probe, relative, parameters, budget)
                save_states(case / f'{name}_states.npz', states)
                threshold = float(spec.task['door_openness_threshold'])
                result['historical_retention'] = retained_grasp_opening(trace, threshold, policy=POLICY)
                result['onset_retention'] = retained_grasp_opening(trace, threshold, policy=FINAL_POLICY)
                result['joint_contact_and_opening'] = joint_metrics(trace, threshold)
            result.update(expert_prefix_steps=prefix_steps, measured_controller_steps=len(trace),
                          mode=args.mode, start=args.start, privileged_expert_pose=True)
            write(case / f'{name}_trace.json', trace)
            results[name] = result
            if not np.array_equal(initial, integration_state(original.env.mj_model, original.env.mj_datas[0])):
                raise AssertionError('Controller changed the immutable parent task')
        receipt = dict(manifest_sha256=digest(args.manifest), source_sha256=digest(source),
                       episode_sha256=digest(fixture / 'episode.json'),
                       code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
        write(case / 'receipt.json', receipt)
        write(case / 'status.json', dict(state='complete', results=results))
        print(json.dumps(dict(case=args.index, results=results), default=lambda x: x.tolist()), flush=True)
    except Exception:
        write(case / 'status.json', dict(state='runtime_or_source_error', error=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=FIXTURE)
    parser.add_argument('--dataset', type=Path, help='Raw-source root containing manifest-relative HDF5 paths')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--mode', choices=['contact', 'opening'], default='contact')
    parser.add_argument('--start', choices=['expert-pregrasp', 'original'], default='expert-pregrasp')
    parser.add_argument('--with-reference', action='store_true')
    parser.add_argument('--max-steps', type=int, default=500, help='Opening action budget including any expert prefix')
    args = parser.parse_args()
    if args.index < 0 or args.index >= len(json.loads(args.manifest.read_text())['cases']):
        parser.error('--index is outside the frozen case list')
    if args.with_reference and args.mode != 'contact':
        parser.error('--with-reference requires --mode contact')
    run_case(args)


if __name__ == '__main__':
    main()
