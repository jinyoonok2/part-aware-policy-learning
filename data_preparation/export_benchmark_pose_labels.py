"""Export the existing early-pose input contract from new expert recordings.

The fixed cohort retains every case. Lack of a pose label never removes a start
from the policy-evaluation population.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from data_preparation.collect_interaction_pose_dataset import inspect_trajectory, transform
from data_preparation.build_early_interaction_pose_view import (
    conditioning_frames, conditioning_records, select_event)
from data_preparation.prepare_test_set import digest, write


def export_case(run, row):
    run = Path(run).resolve()
    folder = run / 'cases' / row['case_id']
    path = folder / 'expert_trajectory.h5'
    with h5py.File(path, 'r') as source:
        if source.attrs.get('format') != 'benchmark_expert_witness_v2':
            raise ValueError('Complete v2 observation recording is required')
        if source.attrs.get('episode_sha256') != row['episode_sha256']:
            raise ValueError('Expert trajectory belongs to another start')
        group = source['traj_0']
        record, target, reason = inspect_trajectory(run,
            dict(minimum_gripper_closure_proxy_m=.005,
                 label_rule='Completed source GRASP_HANDLE with final three grasp flags and measured closure'), row['house'], 0,
            path, group, True)
        status = dict(case_id=row['case_id'], source_format='benchmark_expert_witness_v2',
            pose_label_eligible=False, observations=record['observations'], reason=reason,
            label_rule='Same source phase, closure and early-opening rule as the original Test labels',
            continuous_firm_grasp_certified_by_pose_label=False)
        if target is not None:
            from pose_learning.data import decode
            fractions = [float(np.asarray(decode(x)['opening_percentage']).reshape(-1)[0])
                         for x in group['obs/extra/door_state']]
            events = [*target.get('other_eligible_events', []), target['interaction_event']]
            selected, _ = select_event(events, fractions, maximum=.1, require_no_prior=True)
            if selected is None:
                status['reason'] = 'No eligible initial grasp before any opening above 10%'
            else:
                scene = decode(group['obs_scene'][()])
                frames, _ = conditioning_frames(group['obs/extra/policy_phase'][:],
                    int(scene['policy_phases']['REACH_PRE_GRASP']), selected['start_frame'])
                conditions = conditioning_records(group['obs/extra/robot_base_pose'][:],
                    transform(selected['measured_tcp_world_xyz_wxyz']), frames, fractions)
                target.update(source_split='val', project_role='test', dataset_root=str(run),
                    quality_view='initial_grasp_before_major_opening_v1',
                    h5_sha256=digest(path), interaction_event=selected,
                    other_eligible_events=[event for event in events if event != selected],
                    condition_frames=conditions,
                    selected_source_prior_max_opening_fraction=float(max(fractions[:selected['target_frame']+1])),
                    selected_source_opening_fraction=float(fractions[selected['target_frame']]),
                    expert_reference_origin='new_exact_benchmark_start_expert',
                    replay_status='New physically executed native expert record; not historical source replay',
                    part_relative_label_status='Conditioning-base TCP target; handle-relative force-grasp poses saved separately',
                    event_definition='Earliest eligible completed source GRASP_HANDLE before prior opening exceeds 10%, with source flags and measured closure',
                    label_confidence='Source phase/closure/flag target; force-retention witness is reported separately')
                write(folder / 'pose_target.json', target)
                status.update(pose_label_eligible=True, conditioning_observations=len(conditions), reason=None)
    write(folder / 'pose_label_status.json', status)
    return status


def finalize(run):
    from data_preparation.report_benchmark_expert_verification import expected_protocol, inspect_case
    run = Path(run).resolve()
    study = json.loads((run/'study.json').read_text())
    protocol = expected_protocol(study)
    targets, inventory, counts = [], [], dict(expected_cases=len(study['cases']),
        completed_experts=0, expert_retained_grasp_passes=0, technical_unknown=0, pending=0)
    for row in study['cases']:
        result = inspect_case(run, row, protocol, {})
        classification = result['classification']
        entry = dict(row, expert_classification=classification, pose_label_eligible=False)
        if classification in ('completed_pass', 'completed_fail'):
            counts['completed_experts'] += 1
            counts['expert_retained_grasp_passes'] += int(classification == 'completed_pass')
            folder = run/'cases'/row['case_id']
            status = json.loads((folder/'pose_label_status.json').read_text())
            entry['pose_label_eligible'] = status['pose_label_eligible']
            if status['pose_label_eligible']:
                target = json.loads((folder/'pose_target.json').read_text())
                if digest(run/target['h5']) != target['h5_sha256']:
                    raise ValueError('Generated pose source changed')
                targets.append(target)
        elif classification == 'infrastructure_unknown':
            counts['technical_unknown'] += 1
        else:
            counts['pending'] += 1
        inventory.append(entry)
    complete = counts['completed_experts'] == counts['expected_cases']
    if complete and targets:
        # Validate exactly the same input contract used by the existing trained head.
        temporary = run/'test_early_pose_targets.pending.jsonl'
        temporary.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in targets))
        from pose_learning.data import load_examples
        examples = load_examples(temporary, 'test')
        temporary.replace(run/'test_early_pose_targets.jsonl')
        counts['pose_observations'] = len(examples)
    counts['pose_trajectories'] = len(targets)
    write(run/'collection_inventory.json', dict(cases=inventory, fixed_population_preserved=True))
    write(run/'collection_status.json', dict(state='complete' if complete else 'incomplete', **counts,
        primary_policy_population=len(inventory), physical_failures_do_not_remove_starts=True,
        full_offline_pose_index_ready=bool(complete and targets),
        original_197_unchanged=True, training_used=False))
    print(json.dumps(dict(state='complete' if complete else 'incomplete', **counts), indent=2))
    return counts


def review_pilot(run, size):
    from data_preparation.report_benchmark_expert_verification import expected_protocol, inspect_case
    from pose_learning.data import load_examples, observed_points
    from decord import VideoReader
    run = Path(run).resolve()
    study = json.loads((run/'study.json').read_text())
    if not 1 <= size <= len(study['cases']):
        raise ValueError('Pilot size outside fixed cohort')
    targets = []
    errors = []
    for row in study['cases'][:size]:
        try:
            result = inspect_case(run,row,expected_protocol(study),{})
            if result['classification'] not in ('completed_pass','completed_fail'):
                raise ValueError('Pilot has incomplete or technically unknown expert execution')
            folder=run/'cases'/row['case_id']
            with h5py.File(folder/'expert_trajectory.h5','r') as file:
                group=file['traj_0'];count=len(group['success'])
                if file.attrs.get('format')!='benchmark_expert_witness_v2':
                    raise ValueError('Wrong recording format')
                for key in ('obs/agent/qpos','obs/agent/qvel','obs/time','actions/ee_pose'):
                    if len(group[key])!=count:raise ValueError('State/action/time alignment mismatch')
                for camera in ('head_camera','wrist_camera_l','wrist_camera_r'):
                    if len(VideoReader(str(folder/(camera+'.mp4')),num_threads=1))!=count:
                        raise ValueError('Video/state alignment mismatch')
                    for key in ('intrinsic_cv','extrinsic_cv','cam2world_gl'):
                        values=group['obs/sensor_param/'+camera+'/'+key][:]
                        if len(values)!=count or not np.isfinite(values).all():
                            raise ValueError('Camera parameters are missing or nonfinite')
                for frame in range(count):observed_points(group,frame)
            status=json.loads((folder/'pose_label_status.json').read_text())
            if status['pose_label_eligible']:
                targets.append(json.loads((folder/'pose_target.json').read_text()))
        except (KeyError,ValueError,OSError) as error:
            errors.append(dict(case_id=row['case_id'],error=str(error)))
    if not targets:errors.append(dict(error='No usable source-phase early pose target in the native pilot'))
    observations=0
    if targets and not errors:
        index=run/'pilot_pose_targets.jsonl'
        index.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in targets))
        observations=len(load_examples(index,'test'))
    value=dict(state='passed' if not errors else 'failed',pilot_cases=size,
        usable_pose_trajectories=len(targets),pose_observations=observations,errors=errors,
        membership_changed=False,model_outcomes_used=False,
        gate='Recording compatibility and at least one usable source-phase pose; no force-success-rate threshold')
    write(run/'pilot_review.json',value)
    if errors:raise RuntimeError('Pilot recording/label compatibility failed; see pilot_review.json')
    print(json.dumps(value,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pilot-size', type=int, help='Validate the first N fixed cases before bulk collection')
    args = parser.parse_args()
    if args.pilot_size is not None:review_pilot(args.run,args.pilot_size)
    else:finalize(args.run)


if __name__ == '__main__':
    main()
