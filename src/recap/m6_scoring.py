"""Independent M6 selection scorer. Gold is opened only after the complete run gate."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import EvaluationTarget, Prediction
from .h0 import digest
from .io import now, read_jsonl, sha256, write_json, write_jsonl
from .m2_admin import verify_all_arms
from .scoring import score_claims

ARMS = ('H0', 'H_compact_v1')
TERMINAL_STATUSES = {'ok', 'invalid', 'truncated', 'failed', 'NOT_RUN'}
OPAQUE_INPUT_ID = re.compile(r'^c_[a-f0-9]{32}$')
ROW_INPUT_ID = re.compile(r'(?<!\\)"input_id"\s*:\s*"(c_[a-f0-9]{32})"')


def read_selected_targets(path: Path, selected_ids):
    """Decode only selected opaque IDs from a mixed administrative JSONL file.

    Raw unselected lines are traversed solely to locate their ID. Their JSON,
    label, claim, and other values are never decoded or copied to an artifact.
    """
    selected_ids = set(selected_ids)
    if not selected_ids or any(not OPAQUE_INPUT_ID.fullmatch(cid) for cid in selected_ids):
        raise ValueError('opaque selected input IDs required')
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            identifiers = ROW_INPUT_ID.findall(line)
            if len(identifiers) != 1:
                raise ValueError('mixed target row lacks one unambiguous opaque input ID')
            if identifiers[0] in selected_ids:
                row = json.loads(line)
                if row.get('input_id') != identifiers[0]:
                    raise ValueError('selected target ID mismatch')
                rows.append(row)
    if Counter(row['input_id'] for row in rows) != Counter(selected_ids):
        raise ValueError('selected target row missing or duplicated')
    return rows


def _run_gate(run_dir: Path, arms=ARMS, *, allow_quarantine=False):
    """Validate frozen schedules, every terminal prediction, and all run files."""
    protocol_path = run_dir / 'protocol.json'
    seal_path = run_dir / 'all_arms.seal.json'
    seal = json.loads(seal_path.read_text())
    # Check containment before the reused gate hashes any caller-supplied path.
    for relative in seal.get('files', {}):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or (run_dir / path).is_symlink():
            raise ValueError('unsafe sealed run path')
        if not (run_dir / path).resolve().is_relative_to(run_dir.resolve()):
            raise ValueError('sealed path escapes run directory')
    protocol, seal = verify_all_arms(run_dir, protocol_path)
    arms = tuple(arms)
    if arms not in {ARMS, ('B0', 'B1'), ('V560', 'V1120'), ('C_plan','C_zoom')} or protocol.get('arms') != list(arms):
        raise ValueError('fixed declared comparison arms required')
    order = protocol.get('case_order', [])
    if (not order or len(order) != len(set(order)) or
            any(not isinstance(cid, str) or not cid for cid in order) or
            protocol.get('case_count') != len(order) or len(order) % 2):
        raise ValueError('invalid fixed case denominator')
    expected = {(cid, arm) for cid in order for arm in arms}
    schedule = protocol.get('schedule', [])
    try:
        actual = [(slot['input_id'], slot['arm']) for slot in schedule]
    except (KeyError, TypeError) as error:
        raise ValueError('malformed planned slot schedule') from error
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('every case-arm slot must be planned exactly once')
    rows, prediction_seals = {}, {}
    for arm in arms:
        prediction_path = run_dir / arm / 'predictions.jsonl'
        prediction_seal_path = prediction_path.with_name(prediction_path.name + '.seal.json')
        if str(prediction_seal_path.relative_to(run_dir)) not in seal['files']:
            raise ValueError('prediction seal missing from complete run seal')
        arm_seal = json.loads(prediction_seal_path.read_text())
        arm_rows = read_jsonl(prediction_path)
        if (sha256(prediction_path) != arm_seal.get('prediction_sha256') or
                arm_seal.get('rows') != len(order) or len(arm_rows) != len(order)):
            raise ValueError('prediction seal or planned row denominator mismatch')
        if arm_seal.get('execution_kind') not in {'live', 'mock_contract', 'NOT_RUN'}:
            raise ValueError('invalid prediction execution kind')
        if Counter(row.get('input_id') for row in arm_rows) != Counter(order):
            raise ValueError('missing, duplicate, or unknown terminal prediction slot')
        for row in arm_rows:
            if row.get('status') not in TERMINAL_STATUSES:
                raise ValueError('nonterminal prediction slot')
            prediction = Prediction.model_validate(row)
            case_path = run_dir / arm / 'cases' / (prediction.input_id + '.json')
            state = json.loads(case_path.read_text()).get('execution_state') if case_path.exists() else None
            if prediction.status == 'ok' and state and state.get('evaluation_eligibility') != 'eligible' and not allow_quarantine:
                raise ValueError('committed response quarantined; cannot silently score as eligible')
            if prediction.task_kind != 'claim_verification':
                raise ValueError('wrong prediction task')
            if prediction.status != 'ok' and prediction.answer is not None:
                raise ValueError('failed prediction must not be converted into a label')
        rows[arm], prediction_seals[arm] = arm_rows, arm_seal
    return protocol, seal, rows, prediction_seals


def _paired(rows, arms=ARMS):
    control, challenger = arms
    counts = Counter('both_correct' if row['correct'][control] and row['correct'][challenger]
                     else 'challenger_only' if row['correct'][challenger]
                     else 'incumbent_only' if row['correct'][control] else 'both_wrong'
                     for row in rows)
    return {key: counts[key] for key in ['both_correct', 'challenger_only', 'incumbent_only', 'both_wrong']}


def score_run(run_dir, targets_path, output_path, *, arms=ARMS, visual_processing_verified=False):
    """Score all planned slots once; keep official and incomplete diagnostics distinct."""
    run_dir, targets_path, output_path = map(Path, (run_dir, targets_path, output_path))
    if targets_path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError('targets must stay outside the predictor run directory')
    if output_path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError('score must not mutate the sealed predictor run directory')
    if output_path.exists():
        raise FileExistsError('score output already exists; no silent rescoring')
    arm_names = tuple(arms)
    protocol, seal, rows, prediction_seals = _run_gate(run_dir, arm_names)
    control, challenger = arm_names
    if type(visual_processing_verified) is not bool:
        raise ValueError('visual processing verification must be an explicit boolean')
    if protocol.get('milestone') == 'R4' and (arm_names != ('V560', 'V1120') or protocol.get('normal_control_pair_id') != '0284'):
        raise ValueError('R4 requires its frozen normal control before target access')
    if arm_names == ('V560', 'V1120'):
        control_pair = protocol.get('normal_control_pair_id')
        if (not isinstance(control_pair, str) or not control_pair or
                protocol['case_count'] != (2 if protocol.get('milestone') == 'R4' else 6)):
            raise ValueError('R3 requires six fixed cases and a frozen normal control pair')
    # No target hash/read may move above this completed all-slots gate.
    target_hash = sha256(targets_path)
    if target_hash != protocol.get('targets_sha256'):
        raise ValueError('targets changed after protocol freeze')
    targets = [EvaluationTarget.model_validate(row)
               for row in read_selected_targets(targets_path, protocol['case_order'])]
    if Counter(t.input_id for t in targets) != Counter(protocol['case_order']):
        raise ValueError('target denominator does not match fixed cases')
    pairs = defaultdict(list)
    for target in targets:
        pairs[target.pair_id].append(target)
    if len(pairs) * 2 != protocol['case_count']:
        raise ValueError('fixed complete-pair denominator changed')
    for pair in pairs.values():
        if len({t.work_family for t in pair}) != 1 or pair[0].work_family is None:
            raise ValueError('pair must preserve one known work family')
    if arm_names == ('V560', 'V1120') and control_pair not in pairs:
        raise ValueError('frozen normal control pair absent from selected targets')
    arms, correct = {}, {}
    for arm in arm_names:
        arm_result = score_claims(rows[arm], targets, execution_kind=prediction_seals[arm]['execution_kind'])
        arm_result.update(prediction_sha256=prediction_seals[arm]['prediction_sha256'],
                          terminal_status_counts=dict(Counter(row['status'] for row in rows[arm])),
                          official_score_valid=arm_result['official_status'] == 'COMPUTED')
        arms[arm] = arm_result
        by_id = {row['input_id']: row for row in rows[arm]}
        correct[arm] = {t.input_id: by_id[t.input_id]['status'] == 'ok' and
                       by_id[t.input_id]['answer'] == t.gold_label for t in targets}
    case_table = [{'input_id': t.input_id, 'pair_id': t.pair_id, 'work_family': t.work_family,
                   'gold': t.gold_label, 'correct': {arm: correct[arm][t.input_id] for arm in arm_names},
                   'statuses': {arm: next(r['status'] for r in rows[arm] if r['input_id'] == t.input_id)
                                for arm in arm_names}} for t in targets]
    pair_table = []
    for pair_id, pair in pairs.items():
        correctness = {arm: all(correct[arm][t.input_id] for t in pair) for arm in arm_names}
        pair_table.append({'pair_id': pair_id, 'work_family': pair[0].work_family,
                           'input_ids': [t.input_id for t in pair], 'correct': correctness,
                           'change': 'improved' if correctness[challenger] and not correctness[control]
                           else 'regressed' if correctness[control] and not correctness[challenger]
                           else 'unchanged'})
    complete = all(arms[arm]['official_score_valid'] for arm in arm_names)
    live = all(prediction_seals[arm]['execution_kind'] == 'live' for arm in arm_names)
    a, b = [arms[arm]['all_denominator_diagnostics'] for arm in arm_names]
    improves = b['correct_pairs'] > a['correct_pairs'] and b['correct_cases'] >= a['correct_cases']
    selected = complete and live and improves
    selected_target_path = run_dir.parent / 'scoring' / 'targets.jsonl'
    if selected_target_path.resolve() == targets_path.resolve():
        raise ValueError('selected targets must not overwrite mixed target storage')
    write_jsonl(selected_target_path, [t.model_dump() for t in targets], immutable=True)
    result = {'at': now(), 'route': 'A', 'phase': 'selection',
              'status': 'DONE' if complete else 'PARTIAL',
              'protocol_sha256': sha256(run_dir / 'protocol.json'),
              'all_arms_seal_sha256': sha256(run_dir / 'all_arms.seal.json'),
              'targets_sha256': target_hash,
              'selected_targets_path': str(selected_target_path),
              'selected_targets_sha256': sha256(selected_target_path),
              'target_access': 'selected opaque IDs decoded only after every planned slot was sealed',
              'cases': len(targets), 'pairs': len(pairs),
              'planned_slots': len(protocol['schedule']), 'arms': arms,
              'case_table': case_table, 'pair_table': pair_table,
              'comparison': {'official_comparison_valid': complete,
                             'all_planned_denominators': True,
                             'case_paired': _paired(case_table, arm_names), 'pair_paired': _paired(pair_table, arm_names),
                             'case_correct_difference': b['correct_cases'] - a['correct_cases'],
                             'pair_correct_difference': b['correct_pairs'] - a['correct_pairs'],
                             'small_sample_interpretation': 'descriptive_selection_comparison_only'},
              'selection': {'claim_policy': 'H_compact_v1' if selected else 'H0',
                            'status': 'provisional_development_candidate' if selected else 'incumbent_retained',
                            'rule': 'Both arms complete and live; challenger pair correct strictly increases and case correct does not decrease.',
                            'both_complete_valid': complete, 'both_live': live,
                            'pair_strictly_improved': b['correct_pairs'] > a['correct_pairs'],
                            'case_non_decreased': b['correct_cases'] >= a['correct_cases']},
              'research_goal': 'not_evaluable', 'frontier_comparison': 'NOT_RUN',
              'final_reserve_used': False, 'gold_changed': False,
              'judge_status': 'NOT_RUN', 'tested_commit': None}
    if arm_names == ('B0', 'B1'):
        candidate_status = ('promising_on_selection' if selected else
                            'not_selected_on_selection' if complete else 'inconclusive')
        result.update(route='R2_bounded_accuracy', candidate_status=candidate_status,
                      comparison_control='B0', comparison_candidate='B1',
                      original_incumbent='H0', incumbent_unchanged=True,
                      interpretation='Fixed B1-B0 development comparison under common thinking_token_budget=2048; no global superiority, optimal-budget, or original-H0 accuracy-preservation claim.')
        result['selection'].update(claim_policy='H0', status=candidate_status,
                                   candidate_alias='B1', incumbent_unchanged=True)
        result['comparison']['paired_key_roles'] = {
            'challenger_only': 'B1_only_correct', 'incumbent_only': 'B0_only_correct',
            'both_wrong': 'both_wrong_or_failed'}
    elif arm_names == ('V560', 'V1120'):
        control_ids = [t.input_id for t in pairs[control_pair]]
        regressions = [cid for cid in control_ids if correct[control][cid] and not correct[challenger][cid]]
        case_gain = b['correct_cases'] > a['correct_cases']
        pair_nondecrease = b['correct_pairs'] >= a['correct_pairs']
        equal_scores = b['correct_cases'] == a['correct_cases'] and b['correct_pairs'] == a['correct_pairs']
        if not complete or not live or not visual_processing_verified:
            candidate_status = 'inconclusive'
        elif case_gain and pair_nondecrease and not regressions:
            candidate_status = 'promising_on_visual_challenge'
        elif equal_scores and not regressions:
            candidate_status = 'no_gain_on_visual_challenge'
        else:
            candidate_status = 'tradeoff_not_selected'
        result.update(route='R3_visual_evidence', phase='failure_enriched_development',
                      candidate_status=candidate_status,
                      comparison_control='V560', comparison_candidate='V1120',
                      original_incumbent='H0', incumbent_unchanged=True,
                      operating_default_unchanged=True,
                      visual_processing_verified=visual_processing_verified,
                      visual_processing_verification_basis='Explicit caller assessment after sealing; not inferred by scorer from HTTP status, requested options, or token usage.',
                      normal_control_pair_id=control_pair,
                      normal_control_case_ids=control_ids,
                      normal_control_regression_count=len(regressions),
                      normal_control_regression_ids=regressions,
                      normal_control_regression_definition='V560 correct to V1120 wrong or failed; failures do not become labels.',
                      diagnostic_ceiling=a['correct_cases'] == b['correct_cases'] == 6,
                      metric_scope='Official scorer applied to intentionally failure-enriched development subset; not representative or independent test performance.',
                      interpretation='Same H0 visual-budget comparison only; no general accuracy, optimal-resolution, clinical, frontier, or isolated visual-step causal claim.')
        result['selection'] = {
            'claim_policy': 'H0', 'candidate_alias': 'V1120', 'status': candidate_status,
            'rule': 'All planned outputs valid and live, processor difference verified, V1120 case correct strictly higher, pair correct nondecreased, and no fixed normal-control regression.',
            'both_complete_valid': complete, 'both_live': live,
            'visual_processing_verified': visual_processing_verified,
            'case_strictly_improved': case_gain, 'pair_non_decreased': pair_nondecrease,
            'normal_control_no_regression': not regressions, 'incumbent_unchanged': True}
        result['comparison']['small_sample_interpretation'] = 'intentional_failure_enriched_development_diagnostic_only'
        result['comparison']['paired_key_roles'] = {
            'challenger_only': 'V1120_only_correct', 'incumbent_only': 'V560_only_correct',
            'both_wrong': 'both_wrong_or_failed'}
    if protocol.get('milestone') == 'R4':
        if arm_names != ('V560', 'V1120') or len(pairs) != 1 or control_pair != '0284':
            raise ValueError('R4 must use only the frozen normal control pair0284')
        candidate_status = ('control_check_inconclusive' if not complete or not live else
                            'control_regression_observed' if regressions else
                            'control_check_completed_no_regression_observed')
        result.update(route='R4_control_completion', phase='development_control_reexperiment',
                      candidate_status=candidate_status, diagnostic_ceiling=False,
                      metric_scope='Fresh two-case normal control under one R4 timeout policy; completion recovery is not visual accuracy gain.',
                      interpretation='R3 remains PARTIAL/inconclusive; historical curve and heatmap observations are not rerun. No default promotion or frontier claim.')
        result['selection'] = {'claim_policy':'H0','candidate_alias':'V1120','status':candidate_status,
            'both_complete_valid':complete,'both_live':live,'normal_control_no_regression':not regressions,
            'incumbent_unchanged':True,'rule':'All four valid and live with no V560-correct to V1120-wrong regression; incomplete remains inconclusive.'}
        result['comparison']['small_sample_interpretation']='development_control_reexperiment_only'
    write_json(output_path, result, immutable=True)
    return result
