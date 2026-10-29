"""Allowlisted evidence capabilities, not a claim of OS process isolation."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .calculations import PercentageInput, check_reported_percentage
from .contracts import CaseInput, Contract
from .io import read_jsonl, sha256


class BundleAsset(Contract):
    asset_id: str
    path: str
    media_type: str
    sha256: str


class EvidenceRequest(Contract):
    evidence_id: str


class BundleSource(Contract):
    dataset_id: str
    dataset_revision: str


class EvidenceBundle:
    def __init__(self, root: Path):
        if root.is_symlink():
            raise ValueError("symlink bundle root prohibited")
        self.root = root.resolve()
        source_path = self.root / "source.json"
        if source_path.is_symlink():
            raise ValueError("symlink bundle metadata prohibited")
        self.source = BundleSource.model_validate_json(source_path.read_text()) if source_path.exists() else None
        self.assets = {}
        for row in read_jsonl(self.root / "assets.jsonl"):
            asset = BundleAsset.model_validate(row)
            if asset.asset_id in self.assets or not re.fullmatch(r"a_[0-9a-f]{32}", asset.asset_id):
                raise ValueError("invalid/duplicate opaque asset ID")
            # Only a canonical opaque filename can enter the worker.
            if not re.fullmatch(r"assets/" + asset.asset_id + r"\.(png|jpg|jpeg|pdf|xml|docx|txt)", asset.path):
                raise ValueError("answer-bearing or noncanonical asset path")
            self.assets[asset.asset_id] = asset

    def resolve(self, asset_id: str, case: CaseInput) -> Path:
        if asset_id not in {e.asset_id for e in case.evidence}:
            raise ValueError("asset outside this case")
        asset = self.assets[asset_id]
        relative = Path(asset.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("path escape")
        path = self.root / relative
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("symlink evidence prohibited")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("path escape")
        if case.condition == "pdf_only" and asset.media_type not in {"application/pdf", "image/png", "image/jpeg", "text/plain"}:
            raise ValueError("non-PDF evidence in pdf_only")
        if sha256(path) != asset.sha256:
            raise ValueError("evidence hash mismatch")
        return path

    def observation(self, case: CaseInput) -> dict:
        if case.evidence_condition == "oracle":
            raise ValueError("oracle evidence is excluded from the label-blind runtime")
        for e in case.evidence:
            self.resolve(e.asset_id, case)
        # Lineage remains in local extraction records, never in model observations.
        data = case.model_dump(exclude={"evidence"})
        data["evidence"] = [evidence_observation(e) for e in case.evidence]
        return data


def evidence_observation(evidence) -> dict:
    return evidence.model_dump(exclude={"lineage", "extraction_method"})


class BundleTools:
    def __init__(self, bundle: EvidenceBundle, case: CaseInput):
        self.bundle, self.case = bundle, case
        self.trace = []

    def call(self, name: str, arguments: dict) -> dict:
        if name == "get_evidence":
            args = EvidenceRequest.model_validate(arguments)
            evidence = next((x for x in self.case.evidence if x.evidence_id == args.evidence_id), None)
            if evidence is None:
                raise ValueError("evidence outside this case")
            self.bundle.resolve(evidence.asset_id, self.case)
            result = evidence_observation(evidence)
        elif name == "check_reported_percentage":
            result = check_reported_percentage(PercentageInput.model_validate(arguments))
        else:
            raise ValueError("tool not allowlisted")
        self.trace.append({"tool": name, "arguments": arguments, "result": result})
        return result


def assert_canaries_absent(payload, canaries: list[str]) -> None:
    """Test-time assertion: real gold/canaries are not runtime worker arguments."""
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if any(token in value for token in canaries):
        raise ValueError("outbound payload leaks a test canary")
