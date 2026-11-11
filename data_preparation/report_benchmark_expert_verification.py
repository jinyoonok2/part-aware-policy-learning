"""Report exact-start expert verification without treating missing cases as failures.

Expert witnesses qualify only under the frozen verification configuration.
This report never certifies model/controller equivalence or starts evaluation.
"""

# Support both python -m and frozen source snapshots.
import sys
from pathlib import Path as _BootstrapPath
if __package__ in (None, ""):
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo


PENDING = {'pending', 'queued', 'prepared', 'staged', 'assets_staged',
           'prepared_pending_native_planner_dependencies', 'queued_for_assets'}
ERRORS = {'runtime_or_dependency_error', 'runtime_or_compatibility_error',
          'infrastructure_error', 'asset_error', 'failed', 'error'}
REQUIRED = {'initial_state.npz', 'expert_trajectory.h5', 'left_force_trace.json',
            'right_force_trace.json', 'executed_commands.jsonl',
            'applied_control_substeps.jsonl', 'head_camera.mp4', 'grasp_poses.json',
            'initial_alignment.json', 'resolved_config.json'}


def strict_json(path):
    def reject(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def write(path, value):
    atomic(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def sha256(path, cache=None):
    path = Path(path)
    stat = path.stat()
    fingerprint = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    key = str(path.resolve())
    previous = cache.get(key) if cache is not None else None
    if isinstance(previous, dict) and previous.get('fingerprint') == fingerprint:
        return previous['sha256']
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    result = value.hexdigest()
    if cache is not None:
        cache[key] = dict(fingerprint=fingerprint, sha256=result)
    return result


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Artifact paths must be nonempty relative paths')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its case directory: ' + relative)
    return path


def expected_protocol(study):
    policy = study.get('metric_policy', study.get('retained_grasp_policy'))
    if not isinstance(policy, dict) or not isinstance(policy.get('metric_version'), str):
        raise ValueError('Missing frozen metric policy/version')
    metric_hash = canonical_hash(policy)
    if study.get('metric_policy_sha256', metric_hash) != metric_hash:
        raise ValueError('Study metric-policy hash disagrees with its frozen policy')
    if study.get('metric_version', policy['metric_version']) != policy['metric_version']:
        raise ValueError('Study metric versions disagree')
    horizon = study.get('maximum_executed_actions', study.get('horizon'))
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Missing positive frozen execution budget')
    return policy['metric_version'], metric_hash, horizon


def validate_summary(run, directory, case, summary, protocol, cache):
    version, policy_hash, horizon = protocol
    for key in ('case_id', 'original_episode_index', 'house', 'episode_sha256'):
        if summary.get(key) != case.get(key):
            raise ValueError('Summary ' + key + ' disagrees with frozen case')
    episode = inside(run, case['episode_path'])
    if not valid_hash(case['episode_sha256']) or sha256(episode, cache) != case['episode_sha256']:
        raise ValueError('Frozen benchmark EpisodeSpec is missing or changed')
    if summary.get('metric_version') != version or summary.get('metric_policy_sha256') != policy_hash:
        raise ValueError('Worker metric does not match the frozen protocol')
    steps = summary.get('executed_actions')
    if type(steps) is not int or not 0 <= steps <= horizon or summary.get('horizon') != horizon:
        raise ValueError('Worker execution budget is missing, exceeded or changed')
    if summary.get('model_inference_used') is not False or summary.get('training_started') is not False:
        raise ValueError('Worker must explicitly exclude model inference and training')
    if type(summary.get('retained_grasp_witness_passed')) is not bool:
        raise ValueError('Missing explicit expert witness outcome')
    if summary.get('exact_start_verified') is not True:
        raise ValueError('Exact benchmark initial-state execution was not verified')
    artifacts = summary.get('artifact_sha256', {})
    if not isinstance(artifacts, dict):
        raise ValueError('Artifact hashes must be a mapping')
    for relative, expected in artifacts.items():
        path = inside(directory, relative)
        if not valid_hash(expected) or not path.is_file() or sha256(path, cache) != expected:
            raise ValueError('Witness artifact missing or changed: ' + str(relative))
    passed = summary['retained_grasp_witness_passed']
    if passed or steps > 0:
        if not REQUIRED.issubset(artifacts):
            raise ValueError('Incomplete expert artifact bundle: ' + ', '.join(sorted(REQUIRED - set(artifacts))))
    if passed:
        if summary.get('initial_alignment_passed') is not True:
            raise ValueError('Passing expert witness lacks initial-alignment confirmation')
        if strict_json(directory/'initial_alignment.json').get('passed') is not True:
            raise ValueError('Saved initial alignment does not pass')
        if not strict_json(directory/'grasp_poses.json'):
            raise ValueError('Passing witness has no measured grasp pose')
    return dict(executed_actions=steps, artifact_count=len(artifacts), artifact_sha256=artifacts)


def inspect_case(run, case, protocol, cache):
    case_id = case.get('case_id')
    if not isinstance(case_id, str) or Path(case_id).name != case_id or case_id in ('', '.', '..'):
        raise ValueError('Invalid frozen case ID')
    directory = run/'cases'/case_id
    result = dict(case_id=case_id, original_episode_index=case.get('original_episode_index'),
                  house=case.get('house'), classification='pending', worker_state='pending',
                  physically_executed=False, final_model_test_ready=False)
    summary_path, status_path = directory/'summary.json', directory/'status.json'
    try:
        status = strict_json(status_path) if status_path.exists() else {}
        summary = strict_json(summary_path) if summary_path.exists() else None
        observed = summary if summary is not None else status
        state = observed.get('state', 'pending')
        result['worker_state'] = state
        observed_steps = observed.get('executed_actions', observed.get('step', 0))
        result['physically_executed'] = type(observed_steps) is int and observed_steps > 0
        if state == 'complete':
            if summary is None:
                raise ValueError('Completed worker has no immutable summary.json')
            evidence = validate_summary(run, directory, case, summary, protocol, cache)
            passed = summary['retained_grasp_witness_passed']
            result.update(classification='completed_pass' if passed else 'completed_fail',
                          reference_origin=summary.get('origin', 'unspecified_measured_expert_witness'),
                          retained_grasp_witness_passed=passed,
                          ordinary_opening_ever=summary.get('ordinary_opening_ever'),
                          reason=summary.get('reason', summary.get('feasibility_status')),
                          executed_actions=evidence['executed_actions'],
                          summary_path=str(summary_path.relative_to(run)),
                          summary_sha256=sha256(summary_path, cache),
                          episode_sha256=case['episode_sha256'],
                          metric_version=protocol[0], metric_policy_sha256=protocol[1],
                          artifact_sha256=evidence['artifact_sha256'])
        elif state in PENDING:
            pass
        elif state in ERRORS or 'error' in state or 'failed' in state:
            result.update(classification='infrastructure_unknown',
                          error=observed.get('error', observed.get('reason', state)))
        else:
            result['classification'] = 'running'
            if observed_steps:
                result['observed_step'] = observed_steps
        video = directory/'head_camera.mp4'
        if video.is_file():
            result['video_path'] = str(video.relative_to(run))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(classification='infrastructure_unknown', error=str(error))
    return result


def report(run):
    run = Path(run)
    study = strict_json(run/'study.json')
    protocol = expected_protocol(study)
    cases = study.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('No frozen verification cases')
    ids = [case.get('case_id') for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate frozen case IDs')
    cache_path = run/'audit/report_artifact_hash_cache.json'
    try:
        cache = strict_json(cache_path) if cache_path.exists() else {}
    except (OSError, ValueError):
        cache = {}
    rows = [inspect_case(run, case, protocol, cache) for case in cases]
    counts = Counter(row['classification'] for row in rows)
    finished = counts['completed_pass'] + counts['completed_fail'] + counts['infrastructure_unknown']
    state = ('verification_finished_with_unknowns' if counts['infrastructure_unknown'] else 'verification_complete') if finished == len(cases) else (
        'verification_in_progress' if finished or counts['running'] else 'prepared_expert_verification_pending')
    now = datetime.now(timezone.utc)
    qualified = [dict(case_id=row['case_id'], original_episode_index=row['original_episode_index'],
                      house=row['house'], episode_sha256=row['episode_sha256'],
                      expert_witness_passed=True, reference_origin=row['reference_origin'],
                      summary_path=row['summary_path'], summary_sha256=row['summary_sha256'],
                      artifact_sha256=row['artifact_sha256'],
                      metric_version=protocol[0], metric_policy_sha256=protocol[1],
                      maximum_executed_actions=protocol[2], executed_actions=row['executed_actions'],
                      final_model_test_ready=False,
                      qualification_scope='Frozen native-expert verification configuration only; controller/model-evaluation equivalence is not certified.')
                 for row in rows if row['classification'] == 'completed_pass']
    value = dict(state=state, generated_at_utc=now.isoformat(), expected_cases=len(cases),
                 pending_cases=counts['pending'], running_cases=counts['running'],
                 completed_expert_passes=counts['completed_pass'], completed_expert_failures=counts['completed_fail'],
                 infrastructure_unknown_cases=counts['infrastructure_unknown'],
                 worker_attempted_cases=len(cases)-counts['pending'],
                 physically_executed_cases=sum(row['physically_executed'] for row in rows),
                 qualified_expert_witnesses=len(qualified), metric_version=protocol[0],
                 metric_policy_sha256=protocol[1], maximum_executed_actions=protocol[2],
                 all_workers_attempted=counts['pending']==0,
                 all_cases_physically_executed=sum(row['physically_executed'] for row in rows)==len(cases),
                 verification_all_terminal=finished==len(cases),
                 existing_expert_test_manifest_unchanged=True, baseline_outcomes_used=False,
                 model_evaluation_started=False, training_started=False, final_model_test_ready=False,
                 cases=rows)
    write(run/'expert_qualified_manifest.json', dict(schema_version=1, generated_at_utc=now.isoformat(),
          study_sha256=sha256(run/'study.json', cache), metric_version=protocol[0], metric_policy_sha256=protocol[1],
          qualified_expert_witnesses=len(qualified), final_model_test_ready=False, cases=qualified))
    write(run/'status.json', value)
    write(cache_path, cache)
    stamp = now.astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
    lines = ['# Exact-start benchmark expert verification', '', 'Updated: '+stamp+'.', '',
             f'Planned: {len(cases)} cases. Physically executed: {value["physically_executed_cases"]}.', '',
             '| Expert pass | Completed expert failure | Infrastructure / compatibility unknown | Running | Pending |',
             '|---:|---:|---:|---:|---:|',
             f'| {counts["completed_pass"]} | {counts["completed_fail"]} | {counts["infrastructure_unknown"]} | {counts["running"]} | {counts["pending"]} |', '',
             'Missing cases are pending, not failures. A completed expert failure does not prove the task is impossible. Passing witnesses are bound to this metric and execution configuration; model/controller equivalence and final policy-test readiness are not certified.', '',
             'No MolmoBot inference, training, baseline-outcome filtering, or changes to the existing 197 expert Test trajectories occur here.', '',
             '| Case | House | State | Actions | Ordinary opening | Video |', '|---|---:|---|---:|---|---|']
    html_rows = []
    for row in rows:
        video = f'[video]({quote(row["video_path"])})' if row.get('video_path') else ''
        lines.append(f'| {row["case_id"]} | {row["house"]} | {row["classification"]} | {row.get("executed_actions", row.get("observed_step", ""))} | {row.get("ordinary_opening_ever", "")} | {video} |')
        cells = [html.escape(str(row['case_id'])), html.escape(str(row['house'])),
                 html.escape(row['classification']), html.escape(str(row.get('executed_actions', row.get('observed_step','')))),
                 html.escape(str(row.get('ordinary_opening_ever','')))]
        if row.get('video_path'):
            cells.append(f'<a href="{html.escape(quote(row["video_path"]), quote=True)}">Video</a>')
        else: cells.append('')
        if row.get('error'):
            cells[2] += '<details><summary>Details</summary><pre>'+html.escape(str(row['error']))+'</pre></details>'
        html_rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    atomic(run/'report.md', '\n'.join(lines)+'\n')
    top = '\n'.join(lines[:12])
    atomic(run/'index.html', '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
           '<title>Benchmark expert verification</title><style>body{font:15px system-ui;margin:32px;max-width:1300px}'
           'pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #ccd}'
           'th{background:#eef3f8}details{max-width:700px}</style><h1>Exact-start benchmark expert verification</h1>'
           '<pre>'+html.escape(top)+'</pre><p>Expert passes are provisional execution witnesses; final model-test readiness remains false. Missing cases are pending.</p>'
           '<p><a href="status.json">Structured status</a> · <a href="expert_qualified_manifest.json">Provisional expert witnesses</a> · <a href="report.md">Markdown report</a></p>'
           '<table><thead><tr><th>Case</th><th>House</th><th>State</th><th>Actions</th><th>Ordinary opening</th><th>Video</th></tr></thead><tbody>'
           +''.join(html_rows)+'</tbody></table>')
    print(json.dumps({key:val for key,val in value.items() if key!='cases'}, indent=2))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args.run)
    except Exception as error:
        write(args.run/'status.json', dict(state='report_input_error', error=str(error),
              final_model_test_ready=False, model_evaluation_started=False, training_started=False))
        write(args.run/'expert_qualified_manifest.json', dict(state='invalidated_report_input_error',
              error=str(error), cases=[], final_model_test_ready=False))
        raise


if __name__ == '__main__':
    main()
