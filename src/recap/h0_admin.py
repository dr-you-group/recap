"""M1 administration. Selection uses non-gold metadata; scoring runs only after sealing."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .h0 import PROMPT, digest
from .io import read_jsonl, sha256, write_json


def derive_binding(observation: dict, models: list[dict]):
    """Use observed server/cache/process evidence, never a local model recommendation."""
    settings = observation["settings"]
    model_id, revision, alias = (settings[k] for k in ["MODEL_ID", "MODEL_REVISION", "SERVED_MODEL_NAME"])
    if not observation.get("runtime_ready") or not observation.get("weights") or not all(w["exists"] for w in observation["weights"]):
        raise ValueError("existing runtime/model cache not verified")
    processes = observation.get("vllm_processes", [])
    if len(processes) != 1:
        raise ValueError("expected one identifiable vLLM serving process")
    options = processes[0]["options"]
    if options.get("--revision") != revision or options.get("--served-model-name") != alias:
        raise ValueError("server process and cache revision/alias differ")
    matches = [m for m in models if m.get("id") == alias]
    context = int(settings["MAX_MODEL_LEN"])
    if len(matches) != 1 or matches[0].get("root") != model_id or matches[0].get("max_model_len") != context:
        raise ValueError("endpoint and observed server model differ")
    thinking_lines = observation.get("template_thinking_lines", [])
    switch = (options.get("--reasoning-parser") == "gemma4"
              and any("enable_thinking" in line and "if" in line for line in thinking_lines))
    identity = {"model_id": model_id, "model_revision": revision, "metadata_files": observation["files"],
                "weight_inventory": observation["weights"], "template_sha256": observation["template_sha256"]}
    return {"served_alias": alias, "model_id": model_id, "model_revision": revision,
            "revision_evidence": "observed process arguments and existing cache snapshot directory",
            "max_context": context, "engine_version": observation["engine_version"],
            "template_sha256": observation["template_sha256"], "model_identity_sha256": digest(identity),
            "weight_bytes_sha256_verified": False, "thinking_switch_verified_in_template": switch,
            "server_process_options": options, "internal_effective_settings_verified": False}


def select_smoke(manifest_path: Path, bundle_path: Path, *, seed=20261014):
    rows = read_jsonl(manifest_path)
    # Only these administrative fields are considered. No target/raw annotation is read.
    meta = [{k: row[k] for k in ("input_id", "pair_id", "paper_id")} for row in rows]
    cases = {c["input_id"]: c for c in read_jsonl(bundle_path / "cases.jsonl")}
    assets = {a["asset_id"]: a for a in read_jsonl(bundle_path / "assets.jsonl")}
    pairs = defaultdict(list)
    for row in meta:
        pairs[row["pair_id"]].append(row)
    candidates = []
    for pair_id, members in sorted(pairs.items()):
        if len(members) != 2 or len({m["paper_id"] for m in members}) != 1:
            raise ValueError("smoke selection requires complete same-paper pairs")
        areas, types = [], []
        for m in members:
            case = cases[m["input_id"]]
            asset = assets[case["evidence"][0]["asset_id"]]
            with Image.open(bundle_path / asset["path"]) as image:
                areas.append(image.width * image.height)
            caption = (case.get("caption") or "").lower().strip()
            types.append("table" if caption.startswith("table") else "figure" if caption.startswith(("fig.", "figure")) else "unspecified")
        candidates.append({"pair_id": pair_id, "paper_id": members[0]["paper_id"],
                           "input_ids": sorted(m["input_id"] for m in members),
                           "image_area": max(areas), "type": types[0]})
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Use area quartiles + caption type, selecting a different paper when possible.
    areas = sorted(c["image_area"] for c in candidates)
    for c in candidates:
        c["area_bin"] = min(3, 4 * areas.index(c["image_area"]) // len(areas))
    selected, papers, strata = [], set(), set()
    while len(selected) < 5:
        available = [c for c in candidates if c not in selected and c["paper_id"] not in papers]
        if not available:
            available = [c for c in candidates if c not in selected]
        if not available:
            raise ValueError("fewer than five pairs available")
        choice = max(available, key=lambda c: (c["type"], c["area_bin"]) not in strata)
        selected.append(choice); papers.add(choice["paper_id"]); strata.add((choice["type"], choice["area_bin"]))
    order = [input_id for c in selected for input_id in c["input_ids"]]
    rng.shuffle(order)
    return {"seed": seed, "selection_rule": "seeded metadata-only paper diversity and caption-type/image-area strata",
            "pairs": selected, "case_order": order, "gold_used": False}


def source_snapshot(root: Path):
    paths = [root / name for name in ["pyproject.toml", "uv.lock", ".python-version"]]
    for directory in ["src", "tests", "scripts", "configs"]:
        paths += [p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    return {"files": files, "sha256": digest(files), "tested_commit": None,
            "algorithm": "SHA256(recap.io.canonical(path-to-SHA256 map)), includes terminal newline"}


def freeze_protocol(root: Path, destination: Path, *, binding: dict, profiles: dict,
                    case_order: list[str], stage: str, experiment_path: Path | None = None):
    if stage not in {"smoke", "baseline"} or len(profiles) not in {1, 2}:
        raise ValueError("M1 supports smoke/baseline with at most two profiles")
    if stage == "baseline" and (len(case_order) != 100 or not experiment_path):
        raise ValueError("baseline requires 100 cases and prior written experiment")
    snapshot = source_snapshot(root)
    value = {"stage": stage, "model_binding": binding, "profiles": profiles, "case_order": case_order,
             "cases_sha256": sha256(root / "data/sciclaimeval/bundle/cases.jsonl"),
             "source_files": snapshot["files"], "source_tree_sha256": snapshot["sha256"],
             "h0_prompt": PROMPT, "image_policy": "original_bytes_no_resize_no_crop",
             "timeout_seconds": 300, "network_retries": 2,
             "retry_http_statuses": [429, 500, 502, 503, 504],
             "retry_semantics": "identical payload, no invalid/truncated/empty-answer retry",
             "stop_rules": ["changed binding/hash", "unclear or incomplete attempt", "30 probe/smoke unique", "200 baseline unique", "300 total attempts"],
             "experiment_sha256": sha256(experiment_path) if experiment_path else None,
             "scoring_rule": "seal all profiles before targets; official scores require 100 valid complete predictions"}
    write_json(destination, value, immutable=True)
    write_json(destination.with_suffix(destination.suffix + ".seal.json"), {"sha256": sha256(destination)}, immutable=True)
    return value


def score_sealed_runs(protocol_path: Path, outputs: dict[str, Path], targets_path: Path, destination: Path):
    """Verify every profile seal first, only then open target data."""
    protocol = json.loads(protocol_path.read_text())
    if protocol["stage"] != "baseline" or set(outputs) != set(protocol["profiles"]):
        raise ValueError("all predeclared baseline profiles must be frozen before scoring")
    if sha256(protocol_path) != json.loads(protocol_path.with_suffix(protocol_path.suffix + ".seal.json").read_text())["sha256"]:
        raise ValueError("protocol changed")
    for name, path in outputs.items():
        seal = json.loads((path / "run.seal.json").read_text())
        if digest(seal["files"]) != seal["sha256"]:
            raise ValueError("invalid run seal")
        for rel, expected in seal["files"].items():
            if sha256(path / rel) != expected:
                raise ValueError("frozen artifact changed: " + rel)
    from .scoring import score_files
    targets = {x["input_id"]: x for x in read_jsonl(targets_path)}
    results = {}
    for name, path in outputs.items():
        predictions = read_jsonl(path / "predictions.jsonl")
        if set(p["input_id"] for p in predictions) != set(protocol["case_order"]):
            raise ValueError("scheduled case population differs")
        score = score_files(path / "predictions.jsonl", targets_path)
        confusion, labels, failures = Counter(), Counter(), Counter()
        first_valid, actual_attempts, latencies = 0, 0, []
        for p in predictions:
            if p["status"] == "ok":
                labels[p["answer"]] += 1
                confusion[(targets[p["input_id"]]["gold_label"], p["answer"])] += 1
            else:
                failures[p.get("diagnostic") or p["status"]] += 1
            case_path = path / "cases" / (p["input_id"] + ".json")
            if case_path.exists():
                row = json.loads(case_path.read_text())
                actual_attempts += row["attempt_count"]
                first_valid += p["status"] == "ok" and row["attempt_count"] == 1
                attempts = read_jsonl(path / "attempts" / (p["input_id"] + ".jsonl"))
                latencies += [a["elapsed_seconds"] for a in attempts if a["event"] == "finished"]
        results[name] = {"official_and_coverage": score, "scheduled": len(protocol["case_order"]),
                         "first_attempt_valid": first_valid, "final_valid": sum(p["status"] == "ok" for p in predictions),
                         "generation_attempts": actual_attempts, "failure_breakdown": dict(failures),
                         "predicted_label_counts": dict(labels),
                         "predicted_label_fraction_all_scheduled": {k: v / len(protocol["case_order"]) for k, v in labels.items()},
                         "confusion_matrix_valid_only": [{"gold": a, "predicted": b, "count": n} for (a, b), n in sorted(confusion.items())],
                         "latency_seconds_per_completed_attempt": latencies,
                         "interpretation": "engineering pilot, development-consumed, not a final holdout"}
    write_json(destination, {"protocol_sha256": sha256(protocol_path), "targets_sha256": sha256(targets_path),
                             "profiles": results, "judge": "NOT_RUN"}, immutable=True)
    return results
