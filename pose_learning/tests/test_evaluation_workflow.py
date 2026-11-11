import json
from pathlib import Path
import tempfile
import unittest

from pose_learning.data import digest
from pose_learning.report_policy_evaluation import collect, METHODS
from pose_learning.stage_policy_manifest import freeze_manifest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EvaluationWorkflowTests(unittest.TestCase):
    def test_incomplete_population_cannot_be_frozen_as_full_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'expected.json', dict(cases=[dict(case_id='a'), dict(case_id='b')]))
            write(root/'export.json', dict(cases=[dict(case_id='a')]))
            with self.assertRaises(ValueError):
                freeze_manifest(root/'export.json', root/'expected.json', root/'frozen')
            self.assertFalse((root/'frozen').exists())

    def test_episode_bytes_survive_freezing_and_changed_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'episode.json', dict(seed=42))
            row = dict(case_id='a', episode_path='episode.json', episode_sha256=digest(root/'episode.json'))
            write(root/'expected.json', dict(cases=[dict(case_id='a')]))
            write(root/'export.json', dict(cases=[row], pending_cases=0))
            result = freeze_manifest(root/'export.json', root/'expected.json', root/'frozen')
            self.assertEqual(digest(root/'frozen'/result['cases'][0]['episode_path']), row['episode_sha256'])
            write(root/'episode.json', dict(seed=7))
            with self.assertRaises(ValueError):
                freeze_manifest(root/'export.json', root/'expected.json', root/'changed')

    def test_pending_and_runtime_failure_are_unresolved_not_task_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'manifest.json', dict(cases=[dict(case_id='a'), dict(case_id='b')]))
            write(root/'b/status.json', dict(state='failed_or_incomparable'))
            result = collect(root/'manifest.json', root)
            self.assertEqual(result['counts'], dict(pending=1, technical_failure_or_incomparable=1))
            for method in METHODS:
                metric = result['methods'][method]['retained_grasp_opening_success']
                self.assertEqual(metric['failures'], 0)
                self.assertEqual(metric['unresolved'], 2)
                self.assertIsNone(metric['rate_over_all_requested'])

    def test_paired_recovery_count_and_full_population_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root/'manifest.json', dict(cases=[dict(case_id='a', episode_sha256='episode')]))
            results = {method:dict(state='complete', matched_inputs=True, seed=42,
                horizon=400, metric_policy=dict(version=3), door_opening_success=True,
                retained_grasp_opening_success=method=='predicted_pose') for method in METHODS}
            write(root/'a/case.json', dict(episode_sha256='episode'))
            write(root/'a/status.json', dict(state='complete', paired=True))
            write(root/'a/summary.json', dict(state='complete', case_id='a', results=results))
            result = collect(root/'manifest.json', root)
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['paired']['retained_grasp_opening_success']['predicted_only_success'], 1)
            self.assertEqual(result['methods']['predicted_pose']['retained_grasp_opening_success']['rate_over_all_requested'], 1.)
            results['predicted_pose']['horizon'] = 20
            write(root/'a/summary.json', dict(state='complete', case_id='a', results=results))
            self.assertEqual(collect(root/'manifest.json', root)['counts']['technical_failure_or_incomparable'], 1)


if __name__ == '__main__':
    unittest.main()
