import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_learning.evaluate_policy import oracle_prediction,predicted_world_target
from pose_learning.prepare_oracle_comparison import METHODS,reference_pose,select_references
from pose_learning.report_policy_evaluation import collect


def label(house=1,trajectory='traj_0'):
    return dict(house=house,h5_sha256='h5'+str(house),trajectory=trajectory,
        source_split='val',project_role='test',quality_view='initial_grasp_before_major_opening_v1',
        selected_source_prior_max_opening_fraction=0.,
        interaction_event=dict(measured_tcp_world_xyz_wxyz=[2.,3.,1.,1.,0.,0.,0.],
            measured_tcp_base_xyz_wxyz=[1.,1.,1.,1.,0.,0.,0.],
            robot_base_world_xyz_wxyz=[1.,2.,0.,1.,0.,0.,0.],
            arm='left',target_frame=25,gripper_joint_positions=[.01,-.01]))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class OracleComparisonTests(unittest.TestCase):
    def test_reference_remains_same_world_tcp_when_robot_base_moves(self):
        reference=reference_pose(label(),'a','labels')
        for base in [np.eye(4),np.array([[0.,-1.,0.,5.],[1.,0.,0.,-2.],[0.,0.,1.,0.],[0.,0.,0.,1.]])]:
            prediction=oracle_prediction(reference,base)
            world=predicted_world_target(prediction,base)
            np.testing.assert_allclose(world[:3,3],[2.,3.,1.],atol=1e-7)
            np.testing.assert_allclose(world[:3,:3],np.eye(3),atol=1e-7)
            self.assertEqual(prediction['arm'],'left')
        self.assertFalse(reference['force_certified_grasp'])

    def test_missing_references_are_explicit_with_no_fallback_or_outcome_selection(self):
        rows=[dict(case_id=str(i),house=i,h5_sha256='h5'+str(i),trajectory='traj_0',baseline_success=False) for i in (1,2,3)]
        eligible,missing=select_references(rows,[label(3),label(1)],'labels')
        self.assertEqual([row['case_id'] for row in eligible],['1','3'])
        self.assertEqual([row['case_id'] for row in missing],['2'])
        self.assertEqual([row['original_source_index'] for row in eligible],[0,2])
        with self.assertRaisesRegex(ValueError,'outside'):
            select_references(rows,[label(4)],'labels')
        with self.assertRaisesRegex(ValueError,'Duplicate expert'):
            select_references(rows,[label(1),label(1)],'labels')

    def test_inconsistent_or_late_expert_reference_is_rejected(self):
        value=label();value['interaction_event']['measured_tcp_base_xyz_wxyz'][0]+=.1
        with self.assertRaisesRegex(ValueError,'conventions'):
            reference_pose(value,'a','labels')
        value=label();value['selected_source_prior_max_opening_fraction']=.2
        with self.assertRaisesRegex(ValueError,'prior major opening'):
            reference_pose(value,'a','labels')

    def fixture(self,root):
        reference=reference_pose(label(),'a','labels')
        case=dict(case_id='a',episode_sha256='episode',oracle_pose=reference)
        write(root/'manifest.json',dict(cases=[case],source_requested_cases=2,excluded_missing_reference=1))
        write(root/'a/case.json',case)
        write(root/'a/status.json',dict(state='complete',paired=True))
        results={method:dict(state='complete',matched_inputs=True,seed=42,horizon=400,
            metric_policy=dict(version=3),door_opening_success=method!='baseline',
            retained_grasp_opening_success=method=='oracle_pose',
            oracle_pose_input=method=='oracle_pose',expert_action_input=False) for method in METHODS}
        summary=dict(state='complete',case_id='a',results=results)
        write(root/'a/summary.json',summary)
        return case,summary

    def test_three_method_report_uses_one_matched_denominator_and_exposes_missing_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            report=collect(root/'manifest.json',root,METHODS)
            self.assertEqual(report['counts']['complete_triplets'],1)
            self.assertEqual(report['source_requested_cases'],2)
            self.assertEqual(report['excluded_missing_reference'],1)
            for method in METHODS:
                self.assertEqual(report['methods'][method]['door_opening_success']['measured_pairs'],1)
            self.assertEqual(report['comparisons_to_baseline']['oracle_pose']['retained_grasp_opening_success']['method_only_success'],1)

    def test_partial_triplet_or_oracle_leak_is_unresolved_not_task_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case,summary=self.fixture(root)
            full=copy.deepcopy(summary)
            summary['results'].pop('oracle_pose');write(root/'a/summary.json',summary)
            report=collect(root/'manifest.json',root,METHODS)
            self.assertEqual(report['counts']['technical_failure_or_incomparable'],1)
            self.assertEqual(report['methods']['baseline']['door_opening_success']['failures'],0)
            full['results']['predicted_pose']['oracle_pose_input']=True
            write(root/'a/summary.json',full)
            self.assertEqual(collect(root/'manifest.json',root,METHODS)['counts']['technical_failure_or_incomparable'],1)


if __name__=='__main__':unittest.main()
