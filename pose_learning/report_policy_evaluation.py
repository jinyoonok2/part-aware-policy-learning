"""Report all requested Test starts, keeping pending and technical failures visible."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .data import write_json


METHODS = ('baseline', 'predicted_pose')
METRICS = ('door_opening_success', 'retained_grasp_opening_success')


def collect(manifest, output, methods=METHODS):
    document = json.loads(Path(manifest).read_text())
    cases = document['cases'] if isinstance(document, dict) else document
    methods=tuple(methods)
    if 'baseline' not in methods or len(methods)<2 or len(set(methods))!=len(methods):
        raise ValueError('Unique methods including baseline required')
    complete_key='complete_triplets' if len(methods)==3 else 'complete_pairs'
    identities = [row['case_id'] for row in cases]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('Expected a complete, unique requested source inventory')
    counts = Counter()
    completed, problems = [], []
    for case in cases:
        directory = Path(output) / case['case_id']
        status_path = directory / 'status.json'
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            state = json.loads(status_path.read_text()).get('state') if status_path.exists() else 'pending'
            classification = 'technical_failure_or_incomparable' if state == 'failed_or_incomparable' else 'running' if state == 'running' else 'pending'
            counts[classification] += 1
            if classification == 'technical_failure_or_incomparable':
                problems.append(dict(case_id=case['case_id'], status=json.loads(status_path.read_text())))
            continue
        try:
            summary = json.loads(summary_path.read_text())
            if summary.get('state') != 'complete' or summary['case_id'] != case['case_id']:
                raise ValueError('Incomplete or mismatched paired summary')
            status = json.loads(status_path.read_text())
            if status.get('state') != 'complete' or status.get('paired') is not True:
                raise ValueError('Matched pair did not complete')
            results = summary['results']
            if set(results) != set(methods):
                raise ValueError('All requested methods are required')
            for method in methods:
                result = results[method]
                if result.get('state') != 'complete' or result.get('matched_inputs') is not True:
                    raise ValueError('Unmatched method result')
                if result.get('seed') != 42 or result.get('horizon') != 400:
                    raise ValueError('Result does not use the fixed seed-42, 400-action protocol')
                if any(type(result.get(key)) is not bool for key in METRICS):
                    raise ValueError('Missing measured task outcome')
            if any(results['baseline'].get(key) != results[method].get(key)
                   for method in methods for key in ('seed', 'horizon', 'metric_policy')):
                raise ValueError('Paired settings differ')
            saved_case = json.loads((directory / 'case.json').read_text())
            if case.get('episode_sha256') and saved_case.get('episode_sha256') != case['episode_sha256']:
                raise ValueError('Evaluated EpisodeSpec differs from the requested start')
            if 'oracle_pose' in methods:
                if saved_case.get('oracle_pose') != case.get('oracle_pose') or not case.get('oracle_pose'):
                    raise ValueError('Expert oracle reference changed or is missing')
                for method in methods:
                    if results[method].get('oracle_pose_input') is not (method=='oracle_pose'):
                        raise ValueError('Oracle input must be confined to its declared method')
                    if results[method].get('expert_action_input') is not False:
                        raise ValueError('Expert action replay is not part of this comparison')
            completed.append(summary)
            counts[complete_key] += 1
        except (KeyError, ValueError, TypeError, OSError) as error:
            counts['technical_failure_or_incomparable'] += 1
            problems.append(dict(case_id=case['case_id'], error=str(error)))
    total, known = len(cases), len(completed)
    method_results = {}
    for method in methods:
        method_results[method] = {}
        for metric in METRICS:
            successes = sum(row['results'][method][metric] for row in completed)
            method_results[method][metric] = dict(successes=successes, measured_pairs=known,
                failures=known-successes, unresolved=total-known,
                provisional_rate_among_complete_pairs=successes/known if known else None,
                rate_over_all_requested=successes/total if known == total else None)
    paired = {}
    if 'predicted_pose' in methods:
        for metric in METRICS:
            paired[metric] = dict(
                predicted_only_success=sum(row['results']['predicted_pose'][metric] and not row['results']['baseline'][metric] for row in completed),
                baseline_only_success=sum(row['results']['baseline'][metric] and not row['results']['predicted_pose'][metric] for row in completed))
    comparisons={method:{metric:dict(
        method_only_success=sum(row['results'][method][metric] and not row['results']['baseline'][metric] for row in completed),
        baseline_only_success=sum(row['results']['baseline'][metric] and not row['results'][method][metric] for row in completed))
        for metric in METRICS} for method in methods if method!='baseline'}
    report = dict(state='complete' if known == total else 'incomplete', requested_cases=total,
        counts=dict(counts), methods=method_results, paired=paired, problems=problems,
        comparisons_to_baseline=comparisons,
        scope='Matched policy evaluation of the existing source population; expert feasibility is not certified',
        pending_or_technical_failures_are_not_physical_failures=True,
        partial_rates_are_provisional=True, seeds='Fixed single action seed 42; no checkpoint selection')
    if isinstance(document,dict) and 'source_requested_cases' in document:
        report.update(source_requested_cases=document['source_requested_cases'],
            excluded_missing_reference=document.get('excluded_missing_reference',0),
            scope=document.get('scope',report['scope']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods',nargs='+',choices=('baseline','predicted_pose','oracle_pose'),default=list(METHODS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = collect(args.manifest, args.output,args.methods)
    write_json(args.output / 'report.json', report)
    completed=report['counts'].get('complete_triplets',report['counts'].get('complete_pairs',0))
    lines = ['# Matched Test rollout status', '', f"State: {report['state']}; complete matched groups: {completed}/{report['requested_cases']}.", '',
             'Unfinished or technically incomparable cases remain unresolved. Rates on partial results are provisional.', '',
             '| Method | Ordinary opening | Retained-grasp opening |', '| --- | --- | --- |']
    if 'source_requested_cases' in report:
        lines[4:4]=[f"Source population: {report['source_requested_cases']}; missing expert reference: {report['excluded_missing_reference']}. All method rates use the same eligible cases.",'']
    for method in args.methods:
        values = [report['methods'][method][metric] for metric in METRICS]
        lines.append('| '+method+' | '+' | '.join(f"{value['successes']}/{value['measured_pairs']} measured" for value in values)+' |')
    (args.output / 'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(state=report['state'], requested_cases=report['requested_cases'], counts=report['counts'])))


if __name__ == '__main__':
    main()
