"""Independent scoring: only this layer receives EvaluationTarget."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from .contracts import EvaluationTarget, Prediction
from .io import now, read_jsonl, sha256, write_json, write_jsonl
from .vendor.sciclaimeval_eval import eval_task_1_individual_data, eval_task_1_pair_data


def freeze_predictions(path: Path, predictions: list[Prediction], *, execution_kind: str) -> dict:
    if execution_kind not in {"mock_contract", "live", "NOT_RUN"}:
        raise ValueError("invalid execution kind")
    seal_path = path.with_name(path.name + ".seal.json")
    if path.exists() or seal_path.exists():
        raise FileExistsError("prediction run already exists; use a new run ID")
    write_jsonl(path, [p.model_dump() for p in predictions], immutable=True)
    seal = {"prediction_sha256": sha256(path), "frozen_at": now(),
            "execution_kind": execution_kind, "rows": len(predictions), "feedback_to_predictor": False}
    write_json(seal_path, seal, immutable=True)
    return seal


def score_claims(rows: list[dict], targets: list[EvaluationTarget], *, execution_kind: str) -> dict:
    if not targets or any(t.task_kind != "claim_verification" for t in targets):
        raise ValueError("SciClaimEval scorer requires claim targets")
    truth = {t.input_id: t for t in targets}
    if len(truth) != len(targets):
        raise ValueError("duplicate target IDs")
    pairs = defaultdict(list)
    for t in targets:
        if t.answer_domain != ["Supported", "Refuted"] or t.gold_label not in {"Supported", "Refuted"}:
            raise ValueError("wrong native label domain")
        if not t.pair_id:
            raise ValueError("pair metadata required")
        pairs[t.pair_id].append(t)
    for pair in pairs.values():
        if len(pair) != 2 or {t.gold_label for t in pair} != {"Supported", "Refuted"}:
            raise ValueError("incomplete or malformed target pair")
    frequencies = Counter(row.get("input_id") for row in rows)
    duplicates = [x for x, n in frequencies.items() if n > 1]
    unknown = [x for x in frequencies if x not in truth]
    valid, invalid = {}, []
    for row in rows:
        identifier = row.get("input_id")
        if identifier not in truth or identifier in duplicates:
            continue
        try:
            p = Prediction.model_validate(row)
            if p.task_kind != "claim_verification" or p.status != "ok" or p.answer not in truth[identifier].answer_domain:
                raise ValueError("not a valid completed native prediction")
            valid[identifier] = p.answer
        except (ValidationError, ValueError):
            invalid.append(identifier)
    missing = sorted(set(truth) - set(frequencies))
    correct = sum(answer == truth[i].gold_label for i, answer in valid.items())
    correct_pairs = sum(all(valid.get(t.input_id) == t.gold_label for t in pair) for pair in pairs.values())
    complete = len(valid) == len(truth) and not (duplicates or unknown or invalid or missing)
    result = {"execution_kind": execution_kind,
              "interpretation": "model_evaluation" if execution_kind == "live" else "fixture_or_interface_only_not_model_performance",
              "processing_coverage": {"valid": len(valid), "expected": len(truth), "fraction": len(valid) / len(truth)},
              "failures": {"missing": missing, "invalid": invalid, "duplicate": duplicates, "unknown": unknown},
              "official_status": "COMPUTED" if complete else "NOT_RUN_INCOMPLETE_OR_INVALID",
              "official_metrics": None,
              "all_denominator_diagnostics": {"correct_cases": correct, "total_cases": len(truth),
                 "accuracy": correct / len(truth), "correct_pairs": correct_pairs,
                 "total_pairs": len(pairs), "pair_accuracy": correct_pairs / len(pairs)}}
    if complete:
        gold = [{"claim_id": t.input_id, "label": t.gold_label, "claim_id_pair": t.pair_id} for t in targets]
        result["official_metrics"] = {**eval_task_1_individual_data(valid, gold), **eval_task_1_pair_data(valid, gold)}
    return result


def score_files(prediction_path: Path, target_path: Path) -> dict:
    seal = json.loads(prediction_path.with_name(prediction_path.name + ".seal.json").read_text())
    if sha256(prediction_path) != seal["prediction_sha256"]:
        raise ValueError("prediction changed after freezing")
    rows = read_jsonl(prediction_path)
    if len(rows) != seal["rows"]:
        raise ValueError("prediction row count differs from seal")
    targets = [EvaluationTarget.model_validate(x) for x in read_jsonl(target_path)]
    result = score_claims(rows, targets, execution_kind=seal["execution_kind"])
    return {**result, "prediction_sha256": seal["prediction_sha256"],
            "target_sha256": sha256(target_path), "judge_status": "NOT_RUN"}


def reference_match_diagnostic(*, error_ids: list[str], finding_ids: list[str],
                               matches: list[tuple[str, str]], processing_valid: int,
                               processing_expected: int) -> dict:
    """Contract diagnostic for supplied matches, not an implementation of a judge."""
    if len(set(error_ids)) != len(error_ids) or len(set(finding_ids)) != len(finding_ids):
        raise ValueError("duplicate reference/finding IDs")
    if any(e not in error_ids or f not in finding_ids for e, f in matches):
        raise ValueError("unknown match IDs")
    if not 0 <= processing_valid <= processing_expected:
        raise ValueError("invalid processing denominator")
    hit_errors = {e for e, _ in matches}
    hit_findings = {f for _, f in matches}
    return {"metric_kind": "provided_match_diagnostic_not_official_judge_score",
            "known_error_recovery": len(hit_errors) / len(error_ids) if error_ids else None,
            "reference_matched_finding_precision": len(hit_findings) / len(finding_ids) if finding_ids else None,
            "processing_coverage": processing_valid / processing_expected if processing_expected else None,
            "unmatched_findings": sorted(set(finding_ids) - hit_findings),
            "unmatched_truth_status": "unadjudicated", "actual_false_positive_count": None,
            "judge_status": "NOT_RUN", "official_score": None}
