"""Small observable records. Model assertions never become independent gold."""
from typing import Literal
from pydantic import Field
from .contracts import Contract
from .calculations import PercentageInput, check_reported_percentage
from .h0 import digest

class SourceRef(Contract):
    source_id: str
    location: str = Field(min_length=1, max_length=220)

class Observation(Contract):
    record_id: str
    kind: Literal['layout','row_column','axis','unit','group_legend','value','direction','text','other']
    value: str = Field(max_length=500)
    epistemic_status: Literal['model_observed','text_statement','inferred','unverified']
    sources: list[SourceRef] = Field(min_length=1,max_length=3)

class VisualRecord(Contract):
    scope: str = Field(max_length=500)
    observations: list[Observation] = Field(min_length=3,max_length=16)
    unreadable: list[str] = Field(max_length=6)
    unknowns: list[str] = Field(max_length=6)

class Binding(Contract):
    record_id: str
    claim_atom: str = Field(max_length=400)
    observation_ids: list[str] = Field(max_length=10)
    groups: str | None
    outcome: str | None
    timepoint: str | None
    relation: str | None
    unit: str | None
    reported_statistics: str | None
    assessment: str = Field(max_length=400)
    sources: list[SourceRef] = Field(min_length=1,max_length=3)

class Conflict(Contract):
    record_id: str
    left: SourceRef
    right: SourceRef
    difference: str = Field(max_length=450)

class Revision(Contract):
    record_id: str
    replaces_record_id: str
    new_value: str = Field(max_length=500)
    reason: str = Field(max_length=350)
    sources: list[SourceRef] = Field(min_length=1,max_length=3)

class SourcedNumber(Contract):
    value: str
    sources: list[SourceRef] = Field(max_length=3)

class Premise(Contract):
    asserted: bool | None
    basis: str | None
    sources: list[SourceRef] = Field(max_length=3)

class CalculationRequest(Contract):
    record_id: str
    operation: str
    numerator: SourcedNumber | None
    denominator: SourcedNumber | None
    reported_percent: SourcedNumber | None
    decimal_places: int | None = Field(ge=0,le=10)
    same_population: Premise | None
    same_timepoint: Premise | None
    simple_proportion: Premise | None

class LinkedRecord(Contract):
    observations: list[Observation] = Field(max_length=12)
    bindings: list[Binding] = Field(max_length=8)
    conflicts: list[Conflict] = Field(max_length=6)
    revisions: list[Revision] = Field(max_length=6)
    unknowns: list[str] = Field(max_length=8)
    calculations: list[CalculationRequest] = Field(max_length=3)

class ReviewRecord(Contract):
    record_id: str
    review: str = Field(max_length=2400)
    counterevidence: list[str] = Field(max_length=8)
    unknowns: list[str] = Field(max_length=8)
    calculations: list[CalculationRequest] = Field(max_length=3)

class FinalRecord(Contract):
    answer: Literal['Supported','Refuted']
    evidence_refs: list[str] = Field(min_length=1)
    brief_reason: str = Field(min_length=1,max_length=1800)
    used_record_ids: list[str] = Field(max_length=16)
    verification_summary: str = Field(min_length=1,max_length=1000)


def record_ids(value):
    found=[]
    if isinstance(value,dict):
        if 'record_id' in value: found.append(value['record_id'])
        for x in value.values():found.extend(record_ids(x))
    elif isinstance(value,list):
        for x in value:found.extend(record_ids(x))
    return found


def validate_links(value, sources, previous=(), *, visual=False):
    """Check scope and references, never certify the semantic truth of a record."""
    prior={i for v in previous for i in record_ids(v)}
    ids=record_ids(value)
    observations={o['record_id'] for v in [*previous,value] for o in v.get('observations',[])}
    if len(ids)!=len(set(ids)) or set(ids)&prior:raise ValueError('record ID collision')
    if visual:
        for o in value['observations']:
            if o['epistemic_status'] not in {'model_observed','unverified'}:
                raise ValueError('visual observation cannot assert textual or independently verified facts')
    def walk(x):
        if isinstance(x,dict):
            if 'source_id' in x and x['source_id'] not in sources:raise ValueError('source outside current stage')
            if 'observation_ids' in x and not set(x['observation_ids']) <= observations:raise ValueError('unknown observation ID')
            if 'replaces_record_id' in x and x['replaces_record_id'] not in prior:raise ValueError('revision lacks immutable parent')
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(value)
    if 'used_record_ids' in value and not set(value['used_record_ids'])<=prior:raise ValueError('unknown final record ID')
    return value


def execute_calculations(requests, sources):
    """Only the existing narrow function. Premises remain model-asserted/conditional."""
    results=[]
    for request in requests:
        q=CalculationRequest.model_validate(request)
        result={'status':'not_applicable','reason':'unsupported_operation'}
        if q.operation=='check_reported_percentage':
            numbers=[q.numerator,q.denominator,q.reported_percent]
            premises=[q.same_population,q.same_timepoint,q.simple_proportion]
            usable=lambda refs: bool(refs) and all(r.source_id in sources for r in refs)
            if any(x is None or not usable(x.sources) for x in numbers) or q.decimal_places is None:
                result={'status':'not_applicable','reason':'missing_sourced_inputs'}
            elif any(p is None or p.asserted is not True or not p.basis or not usable(p.sources) for p in premises):
                result={'status':'not_applicable','reason':'unverified_or_missing_conditions'}
            else:
                result=check_reported_percentage(PercentageInput(numerator=q.numerator.value,
                    denominator=q.denominator.value,reported_percent=q.reported_percent.value,
                    decimal_places=q.decimal_places,same_population_verified=True,
                    same_timepoint_verified=True,simple_proportion_verified=True))
        results.append({'request_id':q.record_id,'request_sha256':digest(request),'request':request,
            'function':'recap.calculations.check_reported_percentage' if q.operation=='check_reported_percentage' else None,
            'function_contract_version':'M0_percentage_v1','result':result,
            'premises_independently_verified':False,'conditional_on_model_sourced_inputs':True,
            'source_hashes':sources})
    return results
