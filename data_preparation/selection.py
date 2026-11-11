"""Seeded balanced benchmark candidate selection using metadata only."""
from collections import Counter, defaultdict
import hashlib


def direction(episode):
    text = episode['language']['task_description'].casefold()
    directions = [value for value in ('push', 'pull') if value in text]
    if len(directions) != 1:
        raise ValueError('Expected exactly one door-opening direction')
    return directions[0]


def order_key(seed, kind, value):
    return hashlib.sha256(f'bounded-benchmark-v1:{seed}:{kind}:{value}'.encode()).hexdigest()


def select_bounded_cases(parent_cases, episodes, *, seed=6198, house_count=200,
                         batch_size=100, smoke_episode_index=0,
                         forbidden_houses=()):
    """Return metadata-only selection; no result files or outcome keys are read."""
    if type(house_count) is not int or house_count < 1:
        raise ValueError('Positive house count required')
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError('Batch size must contain complete push/pull pairs')
    if (2 * house_count) % batch_size:
        raise ValueError('Candidate count must fit complete fixed-size batches')
    forbidden = set(forbidden_houses)
    groups = defaultdict(list)
    seen = set()
    smoke = None
    for parent_index, case in enumerate(parent_cases):
        episode_index = case['original_episode_index']
        if episode_index in seen:
            raise ValueError('Duplicate parent benchmark episode identity')
        seen.add(episode_index)
        episode = episodes[episode_index]
        house = case['house']
        if episode['house_index'] != house or episode['data_split'] != 'val':
            raise ValueError('Parent case does not match official Val episode')
        if house in forbidden:
            raise ValueError('Parent Test candidate overlaps a Train house')
        row = dict(parent_case_index=parent_index,
                   case_id=case.get('case_id', f'episode_{episode_index:04d}'),
                   original_episode_index=episode_index, house=house,
                   scene_dataset=episode['scene_dataset'], source_split='val',
                   project_role='test', task_direction=direction(episode),
                   door_body_name=episode['task']['door_body_name'],
                   parent_episode_path=case['episode_path'],
                   episode_sha256=case['episode_sha256'],
                   expert_witness_available=False, grasp_retention_verified=False,
                   qualified_for_model_test=False,
                   prior_exposure_audit_complete=case.get('prior_exposure_audit_complete', False),
                   untouched_test_claim=False)
        groups[house].append(row)
        if episode_index == smoke_episode_index:
            smoke = row
    mixed = [house for house, rows in groups.items()
             if {row['task_direction'] for row in rows} == {'push', 'pull'}]
    if len(mixed) < house_count:
        raise ValueError(f'Only {len(mixed)} mixed houses; {house_count} required')
    ordered_houses = sorted(mixed, key=lambda house: order_key(seed, 'house', house))
    smoke_included = smoke is not None and smoke['house'] in mixed
    if smoke_included:
        ordered_houses.remove(smoke['house'])
        ordered_houses.insert(0, smoke['house'])
    selected_houses = ordered_houses[:house_count]
    selected = []
    repeated_doors = 0
    for house in selected_houses:
        pushes = [row for row in groups[house] if row['task_direction'] == 'push']
        pulls = [row for row in groups[house] if row['task_direction'] == 'pull']
        pairs = [(push, pull) for push in pushes for pull in pulls]
        if smoke_included and house == smoke['house']:
            pairs = [(push, pull) for push, pull in pairs
                     if smoke_episode_index in (push['original_episode_index'], pull['original_episode_index'])]
        push, pull = min(pairs, key=lambda pair: (
            pair[0]['door_body_name'] == pair[1]['door_body_name'],
            order_key(seed, 'pair', f'{house}:{pair[0]["original_episode_index"]}:{pair[1]["original_episode_index"]}')))
        repeated_doors += int(push['door_body_name'] == pull['door_body_name'])
        for row in (push, pull):
            batch_index = len(selected) // batch_size
            selected.append(dict(row, episode_path=row['parent_episode_path'],
                                 planned_case_index=len(selected), batch_index=batch_index,
                                 batch=batch_index + 1,
                                 asset_stage_scope=f'batch_{batch_index + 1:02d}'))
    batches = []
    for begin in range(0, len(selected), batch_size):
        rows = selected[begin:begin + batch_size]
        counts = dict(Counter(row['task_direction'] for row in rows))
        assert counts == {'push': batch_size // 2, 'pull': batch_size // 2}
        assert len({row['house'] for row in rows}) == batch_size // 2
        batches.append(dict(batch_index=begin // batch_size, batch=begin // batch_size + 1,
            asset_stage_scope=f'batch_{begin // batch_size + 1:02d}', candidate_count=len(rows),
            house_count=len(rows) // 2, task_counts=counts,
            planned_case_indices=[row['planned_case_index'] for row in rows],
            case_indices=[row['planned_case_index'] for row in rows],
            parent_case_indices=[row['parent_case_index'] for row in rows],
            case_ids=[row['case_id'] for row in rows],
            original_episode_indices=[row['original_episode_index'] for row in rows],
            house_ids=sorted({row['house'] for row in rows}),
            initial_submission_allowed=begin == 0,
            automatic_submission=False,
            advancement='Review retained/unknown counts before explicitly scheduling the next bounded batch.'))
    smoke_info = dict(original_episode_index=smoke_episode_index,
        parent_case_index=smoke['parent_case_index'] if smoke else None,
        case_id=smoke['case_id'] if smoke else None,
        house=smoke['house'] if smoke else None,
        included_in_candidates=smoke_included,
        counted_toward_candidate_count=smoke_included,
        counted_toward_qualified_target=False,
        reason=('Included before expert outcomes because its house supports a push/pull pair.'
                if smoke_included else 'Integration smoke remains external: its house does not support a push/pull pair in this pool.'))
    return dict(cases=selected, batches=batches, integration_smoke=smoke_info,
        selection_audit=dict(parent_candidates=len(parent_cases), parent_houses=len(groups),
            eligible_mixed_houses=len(mixed), selected_candidates=len(selected),
            selected_houses=len(selected_houses), selected_house_ids=selected_houses,
            task_counts=dict(Counter(row['task_direction'] for row in selected)),
            repeated_target_doors_within_house=repeated_doors,
            selection_seed=seed, model_outcomes_used=False, expert_outcomes_used=False,
            final_qualified_count=0, candidate_count_is_not_a_success_claim=True,
            existing_197_expert_test_trajectories_unchanged=True))

