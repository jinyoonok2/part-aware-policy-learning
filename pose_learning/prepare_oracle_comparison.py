"""Freeze a three-method diagnostic on every available original expert pose.

The existing 197-start evaluation stays untouched. Missing oracle references
are explicit exclusions shared by all three methods in this new diagnostic.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .data import digest,write_json
from .evaluate_policy import load_cases,oracle_prediction

METHODS=('baseline','predicted_pose','oracle_pose')


def reference_key(row):
    return int(row['house']),row['h5_sha256'],row['trajectory']


def reference_pose(row,case_id,label_sha):
    from grasp.runtime import pose_matrix
    if row.get('source_split')!='val' or row.get('project_role')!='test':
        raise ValueError('Oracle reference must belong to the original Test population')
    if row.get('quality_view')!='initial_grasp_before_major_opening_v1':
        raise ValueError('Use the same early-grasp label rule as the trained head')
    if row.get('selected_source_prior_max_opening_fraction',1.)>.1+1e-6:
        raise ValueError('Reference follows prior major opening')
    event=row['interaction_event']
    world=pose_matrix(event['measured_tcp_world_xyz_wxyz'])
    reconstructed=pose_matrix(event['robot_base_world_xyz_wxyz'])@pose_matrix(event['measured_tcp_base_xyz_wxyz'])
    if not np.allclose(world,reconstructed,atol=1e-6,rtol=0):
        raise ValueError('Expert TCP world/base conventions disagree')
    width=float(np.abs(event['gripper_joint_positions']).sum())
    if not np.isfinite(width) or not 0<=width<=.15:
        raise ValueError('Invalid oracle finger-span reference')
    result=dict(source='expert_recorded_initial_grasp_tcp_world',case_id=case_id,
        tcp_world_xyz_wxyz=event['measured_tcp_world_xyz_wxyz'],arm=event['arm'],
        gripper_width_m=width,source_target_frame=event['target_frame'],
        source_h5_sha256=row['h5_sha256'],source_trajectory=row['trajectory'],
        label_table_sha256=label_sha,target_mode='fixed_expert_world_tcp',
        source_prior_max_opening_fraction=row['selected_source_prior_max_opening_fraction'],
        force_certified_grasp=False,
        label_confidence='Measured expert TCP selected by source phase/closure/grasp flags; no force-retention guarantee',
        previous_project_exposure=row.get('previous_project_exposure',False))
    oracle_prediction(result,np.eye(4))
    return result


def select_references(cases,labels,label_sha):
    indexed={}
    for label in labels:
        key=reference_key(label)
        if key in indexed:raise ValueError('Duplicate expert reference identity')
        indexed[key]=label
    requested_keys={reference_key(row) for row in cases}
    if len(requested_keys)!=len(cases):raise ValueError('Duplicate source trajectory identity')
    if set(indexed)-requested_keys:raise ValueError('Oracle label is outside the requested source population')
    eligible,missing=[],[]
    for index,row in enumerate(cases):
        label=indexed.get(reference_key(row))
        if label is None:
            missing.append(dict(case_id=row['case_id'],source_index=index,
                reason='No eligible early expert grasp-pose reference; no model outcome used'))
            continue
        eligible.append(dict(row,original_source_index=index,
            oracle_pose=reference_pose(label,row['case_id'],label_sha)))
    if not eligible:raise ValueError('No oracle references available')
    return eligible,missing


def prepare(source_run,output,project):
    source_run,output,project=Path(source_run).resolve(),Path(output).resolve(),Path(project).resolve()
    if output.exists():raise FileExistsError('Use a fresh oracle comparison run')
    cases=load_cases(source_run/'episodes/policy_episode_manifest.json')
    labels_path=source_run/'metadata/test_early_pose_targets.jsonl'
    labels=[json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    eligible,missing=select_references(cases,labels,digest(labels_path))
    original_plan=json.loads((source_run/'plan.json').read_text())
    checkpoint=Path(original_plan['checkpoint'])
    if digest(checkpoint)!=original_plan['checkpoint_sha256']:
        raise ValueError('The fixed pose checkpoint changed')
    original_hashes=json.loads((source_run/'source_manifest.json').read_text())['files_sha256']
    # Only the evaluation/report code and new oracle helper differ. Bind the
    # predictor and local control implementation to the current comparison.
    protected=[f'pose_learning/{name}.py' for name in ('model','predict','vision','geometry','data')]
    protected += [f'grasp/{name}.py' for name in ('native_adapter','contacts','metrics','runtime','servo')]
    for name in protected:
        if digest(project/name)!=original_hashes[name]:
            raise ValueError('Oracle comparison would change predictor/controller: '+name)
    for folder in ('metadata','episodes','source','logs','policy'):(output/folder).mkdir(parents=True,exist_ok=False)
    shutil.copy2(labels_path,output/'metadata/test_early_pose_targets.jsonl')
    shutil.copy2(source_run/'metadata/source_inventory.json',output/'metadata/source_inventory.json')
    for row in eligible:
        target=output/'episodes'/row['case_id']/'episode.json';target.parent.mkdir()
        shutil.copy2(row['episode_path'],target)
        if digest(target)!=row['episode_sha256']:raise ValueError('Copied episode changed')
        row['episode_path']=str(target)
    manifest=dict(cases=eligible,source_requested_cases=len(cases),excluded_missing_reference=len(missing),
        missing_reference_cases=missing,methods=list(METHODS),source_run=str(source_run),
        scope='Matched expert-reference oracle diagnostic on all labeled starts; references are not force-certified and no upper-bound performance is assumed',
        membership_rule='Every available original early-grasp reference, in unchanged source order; no outcome filtering')
    write_json(output/'episodes/policy_episode_manifest.json',manifest)
    for package in ('pose_learning','grasp'):
        for path in sorted((project/package).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.sh','.json') and '__pycache__' not in path.parts:
                target=output/'source'/path.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
    hashes={str(path.relative_to(output/'source')):digest(path) for path in sorted((output/'source').rglob('*')) if path.is_file()}
    write_json(output/'source_manifest.json',dict(files_sha256=hashes))
    plan=dict(state='prepared',created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(source_run),source_requested_cases=len(cases),eligible_cases=len(eligible),
        missing_oracle_references=len(missing),methods=list(METHODS),expected_rollouts=3*len(eligible),
        checkpoint=str(checkpoint),checkpoint_sha256=original_plan['checkpoint_sha256'],
        oracle_manifest_sha256=digest(output/'episodes/policy_episode_manifest.json'),
        reference_index_sha256=digest(labels_path),seed=42,horizon=400,prediction_interval=8,
        oracle_definition='Fixed measured expert world TCP for the exact training label, re-expressed in the current robot base; same gate/update interval and controller',
        oracle_reference_supplies_arm=True,expert_actions_used=False,force_certified_oracle=False,
        current_197_run_changed=False,protected_code_sha256={name:original_hashes[name] for name in protected},
        model_selected_using_test=False,maximum_simultaneous_oracle_gpus=1)
    write_json(output/'plan.json',plan)
    from .report_policy_evaluation import collect
    write_json(output/'policy/report.json',collect(output/'episodes/policy_episode_manifest.json',output/'policy',METHODS))
    print(json.dumps({key:value for key,value in plan.items() if key!='protected_code_sha256'},indent=2))
    return plan


def verify(run):
    run=Path(run).resolve();plan=json.loads((run/'plan.json').read_text())
    if digest(run/'episodes/policy_episode_manifest.json')!=plan['oracle_manifest_sha256']:
        raise ValueError('Frozen oracle population or reference poses changed')
    if digest(plan['checkpoint'])!=plan['checkpoint_sha256']:raise ValueError('Pose checkpoint changed')
    hashes=json.loads((run/'source_manifest.json').read_text())['files_sha256']
    for name,expected in hashes.items():
        if digest(run/'source'/name)!=expected:raise ValueError('Frozen oracle source changed: '+name)
    print('Frozen oracle source, population, reference poses and checkpoint verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('prepare');create.add_argument('--source-run',type=Path,required=True)
    create.add_argument('--output',type=Path,required=True)
    create.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    check=sub.add_parser('verify');check.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.source_run,args.output,args.project)
    else:verify(args.run)


if __name__=='__main__':main()
