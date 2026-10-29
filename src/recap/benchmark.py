"""The single M0 data adapter: pinned SciClaimEval Subtask 1."""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path

from .contracts import Asset, CaseInput, EvaluationTarget, Evidence
from .download import DownloadCache
from .io import sha256, write_bytes, write_json, write_jsonl

DATA_REVISION = "55ec80b427d9b4eaff8ed9b537a1ea6cb5d665d7"
SCORER_REVISION = "79bb4f73d78c65c043481b6799bdd0810fad3298"
BASE = f"https://huggingface.co/datasets/alabnii/sciclaimeval-shared-task/resolve/{DATA_REVISION}/"
SEED = 20261013
NAMESPACE = uuid.UUID("fa84dd7d-9819-4921-a2d0-1ac61f5b2110")


def opaque(prefix: str, source: str) -> str:
    return prefix + "_" + uuid.uuid5(NAMESPACE, source).hex


def select_pairs(rows: list[dict]) -> tuple[list[list[dict]], dict]:
    if len({x["claim_id"] for x in rows}) != len(rows):
        raise ValueError("duplicate source case IDs")
    groups = defaultdict(list)
    exclusions = Counter()
    for row in rows:
        if row["domain"] != "peerj":
            exclusions["non_peerj"] += 1
        elif row["license_name"] != "CC BY 4.0":
            exclusions["license"] += 1
        elif row["use_context"] not in {"no", "yes"}:
            exclusions["requires_other_context"] += 1
        else:
            groups[row["claim_id_pair"]].append(row)
    complete = {}
    for pair_id, pair in groups.items():
        if pair_id == "no pair" or len(pair) != 2 or {x["label"] for x in pair} != {"Supported", "Refuted"}:
            exclusions["incomplete_pair"] += len(pair)
            continue
        if len({x["paper_id"] for x in pair}) != 1 or len({x["claim"] for x in pair}) != 1:
            raise ValueError("pair paper/claim mismatch")
        complete[pair_id] = sorted(pair, key=lambda x: x["claim_id"])
    by_paper = defaultdict(list)
    for pair_id in sorted(complete):
        by_paper[complete[pair_id][0]["paper_id"]].append(pair_id)
    # Explicit exceptions are intentional: python -O cannot disable these assertions.
    if len(complete) != 66 or len(by_paper) != 27:
        raise ValueError(f"pinned selection assertion failed: {len(complete)} pairs/{len(by_paper)} papers; expected 66/27")
    rng = random.Random(SEED)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    queues = {}
    for paper in papers:
        ids = sorted(by_paper[paper])
        rng.shuffle(ids)
        queues[paper] = deque(ids)
    selected = []
    while len(selected) < 50:
        for paper in papers:
            if queues[paper]:
                selected.append(complete[queues[paper].popleft()])
            if len(selected) == 50:
                break
    if len({p[0]["paper_id"] for p in selected}) < 20:
        raise ValueError("selected paper count below 20")
    return selected, {"raw_cases": len(rows), "eligible_pairs": 66, "eligible_papers": 27,
                      "exclusions": dict(exclusions), "eligible_unselected_pairs": 16}


def input_from_row(row: dict, asset_id: str) -> CaseInput:
    evidence = Evidence(evidence_id=opaque("e", row["claim_id"]), asset_id=asset_id,
                        location_kind="image", location="entire provided image",
                        extraction_method="benchmark_native_image")
    return CaseInput(input_id=opaque("c", row["claim_id"]), task_kind="claim_verification",
                     condition="benchmark_native", claim=row["claim"],
                     answer_domain=["Supported", "Refuted"], caption=row["caption"],
                     context=row["context"] if row["use_context"] == "yes" else None,
                     evidence=[evidence])


def prepare_benchmark(destination: Path, *, offline: bool = False) -> dict:
    destination = destination.resolve()
    raw = DownloadCache(destination / "raw", offline=offline)
    annotation, annotation_info = raw.fetch(BASE + "data/dev_task1_release.json",
                                          "dev_task1_release.json", kind="json")
    pairs, selection = select_pairs(json.loads(annotation.read_bytes()))
    cases, targets, manifest, asset_manifest, bundle_assets = [], [], [], [], []
    for pair in pairs:
        for row in pair:
            remote_path = "data/" + row["evi_path"]
            asset_id = opaque("a", row["evi_path"])
            image, info = raw.fetch(BASE + remote_path, "images/" + row["claim_id"] + ".png", kind="image")
            relative = "assets/" + asset_id + ".png"
            write_bytes(destination / "bundle" / relative, image.read_bytes(), immutable=True)
            case = input_from_row(row, asset_id)
            cases.append(case.model_dump())
            target = EvaluationTarget(input_id=case.input_id, task_kind="claim_verification",
                       gold_label=row["label"], answer_domain=case.answer_domain,
                       annotation_scope="provided claim/evidence pair only",
                       annotation_completeness="complete_for_task", official_scoring_kind="exact",
                       pair_id=row["claim_id_pair"], source_case_id=row["claim_id"],
                       work_family="peerj:" + row["paper_id"],
                       metadata={"operation": row.get("operation"), "detail_others": row.get("detail_others")})
            targets.append(target.model_dump())
            bundle_assets.append({"asset_id": asset_id, "path": relative,
                                  "media_type": "image/png", "sha256": info["sha256"]})
            asset_manifest.append(Asset(asset_id=asset_id, source_url=info["source_url"],
                object_key=remote_path, source_identity={"paper_id": row["paper_id"], "domain": "peerj"},
                version=DATA_REVISION, path=str(image.relative_to(destination)), media_type="image/png",
                sha256=info["sha256"], size=info["size"], retrieved_at=info["retrieved_at"],
                etag=info["etag"], last_modified=info["last_modified"], license=row["license_name"],
                license_evidence=row["license_url"] + " ; pinned annotation license_name",
                rights_status="verified", work_family="peerj:" + row["paper_id"]).model_dump())
            manifest.append({"input_id": case.input_id, "source_case_id": row["claim_id"],
                 "paper_id": row["paper_id"], "pair_id": row["claim_id_pair"],
                 "work_family": "peerj:" + row["paper_id"], "original_split": "dev",
                 "purpose": "engineering_pilot", "development_consumed": True,
                 "license": row["license_name"], "license_url": row["license_url"],
                 "asset_id": asset_id, "asset_sha256": info["sha256"],
                 "source_evidence_path": row["evi_path"], "use_context": row["use_context"]})
    if len(cases) != 100 or len({x["input_id"] for x in cases}) != 100:
        raise ValueError("100 unique cases required")
    write_jsonl(destination / "bundle/cases.jsonl", cases, immutable=True)
    write_jsonl(destination / "bundle/assets.jsonl", bundle_assets, immutable=True)
    write_json(destination / "bundle/source.json", {"dataset_id": "alabnii/sciclaimeval-shared-task",
               "dataset_revision": DATA_REVISION}, immutable=True)
    write_jsonl(destination / "scoring/targets.jsonl", targets, immutable=True)
    write_jsonl(destination / "manifest.jsonl", manifest, immutable=True)
    write_jsonl(destination / "assets.jsonl", asset_manifest, immutable=True)
    summary = {**selection, "dataset_revision": DATA_REVISION, "scorer_revision": SCORER_REVISION,
               "seed": SEED, "selected_pairs": len(pairs), "selected_cases": len(cases),
               "selected_papers": len({x["paper_id"] for x in manifest}),
               "source_annotation_sha256": annotation_info["sha256"],
               "manifest_sha256": sha256(destination / "manifest.jsonl"),
               "cases_sha256": sha256(destination / "bundle/cases.jsonl"),
               "targets_sha256": sha256(destination / "scoring/targets.jsonl"),
               "asset_count": len(bundle_assets), "asset_bytes": sum(x["size"] for x in asset_manifest),
               "original_split": "dev", "development_consumed": True,
               "live_inference": "NOT_RUN", "official_model_score": None}
    write_json(destination / "summary.json", summary, immutable=True)
    return {**summary, "cache_hits_this_run": raw.hits, "downloads_this_run": raw.downloads}
