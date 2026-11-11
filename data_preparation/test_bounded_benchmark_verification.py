"""Selection checks without expert execution or mutable experiment outcomes."""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import copy
import unittest

from data_preparation.selection import select_bounded_cases


def fixture(houses=6, pure_smoke=False):
    episodes, cases = [], []
    if pure_smoke:
        episodes.append(dict(house_index=0, data_split='val', scene_dataset='procthor-10k',
            language=dict(task_description='Pull the door open'), task=dict(door_body_name='zero')))
        cases.append(dict(house=0, original_episode_index=0, episode_path='cases/episode_0000/episode.json',
            episode_sha256='0'*64, case_id='episode_0000'))
    for house in range(1 if pure_smoke else 0, houses + (1 if pure_smoke else 0)):
        for task, door in [('push','a'), ('pull','a'), ('pull','b')]:
            index = len(episodes)
            episodes.append(dict(house_index=house, data_split='val', scene_dataset='procthor-10k',
                language=dict(task_description=f'{task.title()} the door open'),
                task=dict(door_body_name=f'house{house}_{door}')))
            cases.append(dict(house=house, original_episode_index=index, episode_path=f'cases/{index}/episode.json',
                episode_sha256=f'{index:064x}', case_id=f'episode_{index:04d}'))
    return cases, episodes


class BoundedSelectionChecks(unittest.TestCase):
    def select(self, cases, episodes, **kwargs):
        return select_bounded_cases(cases, episodes, house_count=4, batch_size=4, **kwargs)

    def test_house_direction_balance_and_complete_batch_pairs(self):
        cases, episodes = fixture()
        result = self.select(cases, episodes)
        self.assertEqual(len(result['cases']), 8)
        self.assertEqual(len({row['house'] for row in result['cases']}), 4)
        self.assertEqual(result['selection_audit']['task_counts'], {'push':4,'pull':4})
        self.assertEqual(result['selection_audit']['repeated_target_doors_within_house'], 0)
        for batch in result['batches']:
            self.assertEqual(batch['house_count'], 2)
            self.assertEqual(batch['task_counts'], {'push':2,'pull':2})
            self.assertFalse(batch['automatic_submission'])
        self.assertTrue(result['batches'][0]['initial_submission_allowed'])
        self.assertFalse(result['batches'][1]['initial_submission_allowed'])

    def test_outcome_changes_do_not_change_selection_or_first_batch(self):
        cases, episodes = fixture()
        before = self.select(cases, episodes)
        changed = copy.deepcopy(cases)
        for index, case in enumerate(changed):
            case.update(baseline_success=index%2==0, retained_grasp_witness_passed=index%3==0,
                        expert_score=1000-index, state='complete')
        after = self.select(changed, episodes)
        self.assertEqual(before, after)

    def test_seed_reproduces_and_changes_order_without_changing_balance(self):
        cases, episodes = fixture(12, pure_smoke=True)
        first = self.select(cases, episodes, seed=6198)
        self.assertEqual(first, self.select(cases, episodes, seed=6198))
        second = self.select(cases, episodes, seed=999)
        self.assertNotEqual(first['selection_audit']['selected_house_ids'], second['selection_audit']['selected_house_ids'])
        self.assertEqual(first['selection_audit']['task_counts'], second['selection_audit']['task_counts'])

    def test_pure_pull_integration_smoke_is_external_to_test_candidates(self):
        cases, episodes = fixture(pure_smoke=True)
        result = self.select(cases, episodes)
        self.assertFalse(result['integration_smoke']['included_in_candidates'])
        self.assertFalse(result['integration_smoke']['counted_toward_qualified_target'])
        self.assertNotIn(0,{row['house'] for row in result['cases']})

    def test_mixed_smoke_is_included_without_using_outcomes(self):
        cases, episodes = fixture()
        result = self.select(cases, episodes)
        self.assertTrue(result['integration_smoke']['included_in_candidates'])
        self.assertIn(0,{row['original_episode_index'] for row in result['cases']})
        self.assertEqual(result['cases'][0]['house'],0)

    def test_train_overlap_and_duplicate_episode_identity_fail_closed(self):
        cases, episodes = fixture()
        with self.assertRaisesRegex(ValueError,'Train house'):
            self.select(cases, episodes, forbidden_houses=[5])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.select(cases+[copy.deepcopy(cases[0])], episodes)


if __name__ == '__main__':
    unittest.main()
