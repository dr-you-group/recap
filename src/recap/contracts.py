"""Small, explicit data contracts. Prediction types contain no gold metadata."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskKind = Literal["claim_verification", "document_audit"]
Condition = Literal["benchmark_native", "pdf_only", "docx_native", "xml_assisted"]
Support = Literal["supported", "unsupported", "unknown"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Asset(Contract):
    asset_id: str
    source_url: str
    object_key: str | None = None
    source_identity: dict[str, str | int | None]
    version: str
    path: str
    media_type: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    retrieved_at: str
    source_md5: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    license: str
    license_evidence: str
    rights_status: Literal["verified", "unresolved"] = "unresolved"
    state: Literal["available", "unavailable", "version_unverified"] = "available"
    work_family: str | None = None
    parent_sha256: str | None = None
    changes: str | None = None


class Evidence(Contract):
    evidence_id: str = Field(pattern=r"^e_[a-f0-9]{32}$")
    asset_id: str = Field(pattern=r"^a_[a-f0-9]{32}$")
    text: str | None = None
    raw_value: str | None = None
    unit: str | None = None
    normalized_value: str | float | None = None
    location_kind: Literal["page", "jats_xpath", "docx_cell", "docx_paragraph", "image", "provided"]
    location: str
    extraction_method: str
    lineage: list[str] = Field(default_factory=list)
    epistemic_status: Literal["observed", "inferred", "calculated"] = "observed"
    analysis_population: str | None = None
    group: str | None = None
    comparator: str | None = None
    outcome: str | None = None
    timepoint: str | None = None
    adjustment: str | None = None
    effect_measure: str | None = None


class CaseInput(Contract):
    input_id: str = Field(pattern=r"^c_[a-f0-9]{32}$")
    task_kind: TaskKind
    condition: Condition
    evidence_condition: Literal["provided", "agent-selected", "oracle"] = "provided"
    claim: str | None = None
    answer_domain: list[str] | None = None
    review_scope: str | None = None
    evidence: list[Evidence]
    caption: str | None = None
    context: str | None = None

    @model_validator(mode="after")
    def task_contract(self):
        if self.task_kind == "claim_verification":
            if not self.claim or not self.claim.strip() or not self.answer_domain:
                raise ValueError("claim_verification requires claim and native answer_domain")
            if len(set(self.answer_domain)) != len(self.answer_domain):
                raise ValueError("duplicate answer_domain")
        elif not self.review_scope or self.answer_domain is not None:
            raise ValueError("document_audit requires review_scope and no answer_domain")
        if not self.evidence:
            raise ValueError("at least one manuscript/evidence reference is required")
        if len({x.evidence_id for x in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence IDs")
        if self.condition == "pdf_only" and any(x.location_kind == "jats_xpath" for x in self.evidence):
            raise ValueError("JATS evidence prohibited in pdf_only")
        return self


class KnownError(Contract):
    error_id: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    origin: Literal["human_injected", "model_injected", "natural", "fixture", "unknown"]


class EvaluationTarget(Contract):
    input_id: str
    task_kind: TaskKind
    gold_label: str | None = None
    answer_domain: list[str] | None = None
    known_errors: list[KnownError] | None = None
    gold_evidence: list[str] = Field(default_factory=list)
    annotation_scope: str
    annotation_completeness: Literal["complete_for_task", "known_errors_only", "unknown"]
    official_scoring_kind: Literal["exact", "lexical", "learned", "llm_judge", "hybrid", "none"]
    pair_id: str | None = None
    source_case_id: str | None = None
    work_family: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def target_contract(self):
        if self.task_kind == "claim_verification":
            if not self.answer_domain or self.gold_label not in self.answer_domain or self.known_errors is not None:
                raise ValueError("claim target must preserve its native labels")
        else:
            if self.gold_label is not None or self.known_errors is None:
                raise ValueError("audit target requires known_errors, not gold_label")
            if self.annotation_completeness == "complete_for_task":
                raise ValueError("M0 has no complete whole-document gold")
        return self


class Finding(Contract):
    finding_id: str
    evidence_refs: list[str] = Field(min_length=1)
    asserted_error: str = Field(min_length=1)
    status: Literal["asserted", "uncertain", "not_applicable"]


class Prediction(Contract):
    input_id: str
    task_kind: TaskKind
    answer: str | None = None
    findings: list[Finding] | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["ok", "invalid", "truncated", "failed", "NOT_RUN"]
    diagnostic: str | None = None

    @model_validator(mode="after")
    def output_contract(self):
        if self.task_kind == "claim_verification" and self.findings is not None:
            raise ValueError("claim output uses answer")
        if self.task_kind == "document_audit" and self.answer is not None:
            raise ValueError("audit output uses findings")
        if self.status == "ok":
            if self.task_kind == "claim_verification" and self.answer is None:
                raise ValueError("successful claim output needs answer")
            if self.task_kind == "document_audit" and self.findings is None:
                raise ValueError("successful audit output needs findings (possibly empty)")
        return self


class CapabilityProfile(Contract):
    backend: str
    transport: str
    model_id: str
    model_revision: str | None
    image: Support = "unknown"
    tools: Support = "unknown"
    structured_output: Support = "unknown"
    reasoning: Support = "unknown"
    sampling: Support = "unknown"
    logprobs: Support = "unknown"
    evidence: dict[str, str] = Field(default_factory=dict)
    reasoning_wire_key: str | None = None
    max_images: int = Field(default=2, ge=0)
    image_policy: Literal["original_bytes_no_resize"] = "original_bytes_no_resize"
    serving_verified: bool = False


class GenerationOptions(Contract):
    output_budget: int = Field(default=512, gt=0)
    reasoning: Literal["enabled", "disabled"] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    structured_output: bool = False
    logprobs: bool = False
    use_tools: bool = False


class RunRecord(Contract):
    run_id: str
    input_id: str
    execution_kind: Literal["mock_contract", "live", "NOT_RUN"]
    profile: CapabilityProfile
    requested: dict[str, Any]
    transmitted: dict[str, Any]
    observed: dict[str, Any]
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    status: str
    retries: int = 0
    code_version: str
    input_sha256: str
    input_asset_sha256: dict[str, str] = Field(default_factory=dict)
    dataset_id: str | None = None
    dataset_revision: str | None = None
    internal_effective_settings_verified: bool = False
