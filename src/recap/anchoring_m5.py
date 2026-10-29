"""Conservative raw-text anchoring and item-isolated M5 discovery recovery.

Normalization only collapses the declared whitespace allowlist. Offsets index
Python characters in preserved raw source text; they are never PDF coordinates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .document_review import Candidate, Discovery, SummaryItem
from .h0 import digest
from .io import now, sha256, write_json

RULE_VERSION = 'M5_whitespace_anchor_v2'
WHITESPACE = frozenset(' \t\r\n\f\u00a0')


class SourceBoundaryError(ValueError):
    """A forbidden source reference is a hard failure, never an isolated omission."""


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize_with_mapping(raw: str):
    """Return normalized text plus one raw half-open span per normalized character."""
    if not isinstance(raw,str):raise TypeError('source/quote text must be strings')
    chars=[];spans=[];i=0
    while i<len(raw):
        start=i
        if raw[i] in WHITESPACE:
            while i<len(raw) and raw[i] in WHITESPACE:i+=1
            chars.append(' ');spans.append((start,i))
        else:
            chars.append(raw[i]);i+=1;spans.append((start,i))
    return ''.join(chars),spans


def _positions(haystack,needle):
    start=0
    while True:
        pos=haystack.find(needle,start)
        if pos<0:return
        yield pos,pos+len(needle)
        start=pos+1


def anchor_quote(quote, source:dict, *, prefix:str|None=None, suffix:str|None=None,
                 raw_span:tuple[int,int]|None=None)->dict:
    if hasattr(quote,'model_dump'):quote=quote.model_dump()
    if isinstance(quote,dict):
        if quote.get('source_id')!=source.get('source_id'):
            raise SourceBoundaryError('quote source is outside its supplied source')
        raw_quote=quote.get('quote')
    else:raw_quote=quote
    if not isinstance(raw_quote,str):raise ValueError('quote must be a string')
    if not isinstance(source.get('source_id'),str) or not isinstance(source.get('text'),str):
        raise SourceBoundaryError('invalid host source record')
    raw=source['text'];source_id=source['source_id']
    record={'raw_quote':raw_quote,'raw_excerpt':None,'source_id':source_id,
            'source_sha256':_hash(raw),'source_hash_scope':'UTF-8 bytes of this exact allowed source text',
            'source_asset_sha256':source.get('source_asset_sha256'),
            'source_location':source.get('location'),'raw_start':None,'raw_end':None,
            'offset_unit':'Python Unicode character index; end exclusive; not PDF bbox',
            'match_method':None,'rule_version':RULE_VERSION,'ambiguity':False,
            'normalization_reversible':False,'mapping_basis':'Preserved raw text and per-character raw spans',
            'verification_scope':'Source presence only; not value extraction, population binding or claim truth'}
    if not raw_quote or all(c in WHITESPACE for c in raw_quote):
        return {**record,'status':'rejected','reason':'empty_or_whitespace_only_quote','occurrence_count':0}
    spans=list(_positions(raw,raw_quote));method='exact'
    if not spans:
        normalized,mapping=normalize_with_mapping(raw)
        qnorm,_=normalize_with_mapping(raw_quote)
        spans=[(mapping[start][0],mapping[end-1][1]) for start,end in _positions(normalized,qnorm)]
        method='allowlisted_whitespace'
    record['match_method']=method if spans else None
    record['occurrence_count']=len(spans)
    if not spans:return {**record,'status':'rejected','reason':'quote_not_found_without_forbidden_changes'}
    if raw_span is not None:
        if (not isinstance(raw_span,(tuple,list)) or len(raw_span)!=2
                or any(type(x) is not int for x in raw_span)):
            raise ValueError('raw_span must contain two actual character offsets')
        spans=[s for s in spans if s==tuple(raw_span)]
    if prefix is not None:
        if not isinstance(prefix,str) or not prefix:raise ValueError('prefix must be nonempty actual source text')
        spans=[s for s in spans if raw[:s[0]].endswith(prefix)]
    if suffix is not None:
        if not isinstance(suffix,str) or not suffix:raise ValueError('suffix must be nonempty actual source text')
        spans=[s for s in spans if raw[s[1]:].startswith(suffix)]
    record['disambiguation']={'raw_span':list(raw_span) if raw_span is not None else None,
                              'prefix':prefix,'suffix':suffix,'remaining_occurrences':len(spans)}
    if not spans:return {**record,'status':'rejected','reason':'supplied_location_or_context_does_not_match'}
    if len(spans)!=1:
        return {**record,'status':'ambiguous','ambiguity':True,'reason':'multiple_source_occurrences',
                'candidate_raw_spans':[list(s) for s in spans]}
    start,end=spans[0]
    return {**record,'status':'accepted','raw_excerpt':raw[start:end],
            'raw_start':start,'raw_end':end,'ambiguity':False,'reason':None}


def _top_level(raw):
    if isinstance(raw,str):
        try:value=json.loads(raw)
        except json.JSONDecodeError as error:raise ValueError('malformed top-level JSON') from error
    elif isinstance(raw,dict):value=raw
    else:raise ValueError('malformed top-level discovery')
    if not isinstance(value,dict) or set(value)!={'summary','candidates','unverified'}:
        raise ValueError('malformed top-level discovery fields')
    for field,limit in [('summary',12),('candidates',40),('unverified',10)]:
        if not isinstance(value[field],list) or len(value[field])>limit:
            raise ValueError('malformed top-level discovery collection')
    if any(not isinstance(x,str) for x in value['unverified']):
        raise ValueError('malformed top-level unverified items')
    return value


def recover_discovery(raw, sources, allowed_source_ids=None)->dict:
    """Recover independent items while retaining every failed item's original denominator."""
    value=_top_level(raw)
    if isinstance(sources,dict):
        # An index is accepted as convenience, but the caller must restrict its sources
        # to those actually supplied during discovery (not later visual/full-page inputs).
        source_rows=sources['sources'] if 'sources' in sources else list(sources.values())
    else:source_rows=list(sources)
    lookup={s['source_id']:s for s in source_rows}
    if len(lookup)!=len(source_rows):raise SourceBoundaryError('duplicate allowed source IDs')
    allowed=set(lookup) if allowed_source_ids is None else set(allowed_source_ids)
    if not allowed<=set(lookup):raise SourceBoundaryError('allowed source missing from supplied index')
    def inspect_refs(obj):
        if isinstance(obj,dict):
            if 'source_id' in obj and (not isinstance(obj['source_id'],str) or obj['source_id'] not in allowed):
                raise SourceBoundaryError('forbidden or unknown source ID in discovery')
            for child in obj.values():inspect_refs(child)
        elif isinstance(obj,list):
            for child in obj:inspect_refs(child)
    inspect_refs(value)
    candidate_ids=[c.get('candidate_id') for c in value['candidates'] if isinstance(c,dict) and isinstance(c.get('candidate_id'),str)]
    if len(candidate_ids)!=len(set(candidate_ids)):raise ValueError('duplicate candidate IDs are a hard contract failure')
    buckets={state:{'summary':[],'candidates':[]} for state in ['accepted','rejected','ambiguous']}
    all_anchors=[];accepted={'summary':[],'candidates':[],'unverified':list(value['unverified'])}
    for group,cls in [('summary',SummaryItem),('candidates',Candidate)]:
        for index,item in enumerate(value[group]):
            entry={'index':index,'item':item,'anchors':[]}
            try:parsed=cls.model_validate(item,strict=True)
            except ValidationError as error:
                entry['reason']='item_schema_invalid'
                entry['schema_errors']=[{'location':list(e['loc']),'type':e['type']} for e in error.errors()]
                buckets['rejected'][group].append(entry);continue
            if group=='summary' and parsed.value is not None and not parsed.sources:
                entry['reason']='non_null_summary_without_source';buckets['rejected'][group].append(entry);continue
            for qindex,quote in enumerate(parsed.sources):
                anchor=anchor_quote(quote,lookup[quote.source_id])
                tagged={**anchor,'group':group,'item_index':index,'source_ref_index':qindex}
                entry['anchors'].append(tagged);all_anchors.append(tagged)
            statuses={a['status'] for a in entry['anchors']}
            state='rejected' if 'rejected' in statuses else 'ambiguous' if 'ambiguous' in statuses else 'accepted'
            entry['reason']=None if state=='accepted' else 'one_or_more_required_quotes_'+state
            buckets[state][group].append(entry)
            if state=='accepted':accepted[group].append(parsed.model_dump())
    Discovery.model_validate(accepted)
    counts={group:{'original':len(value[group]),**{state:len(buckets[state][group]) for state in buckets}}
            for group in ['summary','candidates']}
    return {'rule_version':RULE_VERSION,'discovery':accepted,**buckets,'original_counts':counts,'anchors':all_anchors,
            'all_required_quotes_retained':True,'raw_quotes_modified':False,
            'candidate_recall':'unknown; item isolation does not discover additional manuscript claims',
            'source_presence_is_semantic_validation':False}


def _verify_parent_review(path):
    seal=json.loads((path/'review.seal.json').read_text())
    for rel,expected in seal['files'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts:raise SourceBoundaryError('unsafe parent seal path')
        if sha256(path/p)!=expected:raise ValueError('M4 parent review seal mismatch')
    if digest(seal['files'])!=seal['sha256']:raise ValueError('M4 parent seal aggregate mismatch')
    return seal


def replay_m4_discovery(root:Path,output:Path|None=None)->dict:
    """One CPU-only replay of the already-sealed M4 PDF discovery; no model calls."""
    root=Path(root);parent=root/'artifacts/m4/pdf_review';attempts=root/'artifacts/private/m4/review_pdf_review.attempts'
    _verify_parent_review(parent)
    selection=json.loads((parent/'selection.json').read_text());index=json.loads((parent/'source_index.json').read_text())
    if index['condition']!='pdf_only':raise SourceBoundaryError('M4 replay condition changed')
    original_file=Path(index.get('local_path',index['source_asset']['path']))
    if not original_file.is_absolute():original_file=root/original_file
    if sha256(original_file)!=index['source_asset']['sha256']:raise ValueError('M4 original PDF hash changed')
    response_path=attempts/'discovery.response.json';response=json.loads(response_path.read_text())
    events_path=attempts/'discovery.events.jsonl'
    if sha256(events_path)!=response['events_sha256']:raise ValueError('M4 response event hash mismatch')
    finished=[]
    for line in events_path.open():
        event=json.loads(line)
        if event.get('event')=='finished':finished.append(event)
    if len(finished)!=1:raise ValueError('expected one original M4 terminal record')
    terminal=finished[0]
    if not terminal.get('done_received') or terminal.get('finish_reason')!='stop' or response['finish_reason']!='stop':
        raise ValueError('incomplete transport cannot be recovered as final JSON')
    raw=terminal['partial_content']
    selected=[{**s,'source_asset_sha256':index['source_asset']['sha256']} for s in selection['sources']]
    recovered=recover_discovery(raw,selected)
    result={'at':now(),'run_kind':'M5_derived_CPU_discovery_replay','fresh_discovery':False,
            'replayed_discovery':True,'new_model_generations':0,'M4_status_changed':False,
            'whole_document_new_end_to_end':False,'parent':{
                'review_path':str(parent.relative_to(root)),'review_seal_sha256':sha256(parent/'review.seal.json'),
                'source_index_sha256':sha256(parent/'source_index.json'),'source_selection_sha256':sha256(parent/'selection.json'),
                'response_path':str(response_path.relative_to(root)),'response_sha256':sha256(response_path),
                'events_path':str(events_path.relative_to(root)),'events_sha256':response['events_sha256'],
                'raw_discovery_sha256':_hash(raw),'original_asset_sha256':index['source_asset']['sha256']},
            'original_discovery_source_ids':[s['source_id'] for s in selected],
            'original_discovery_source_hashes':{s['source_id']:_hash(s['text']) for s in selected},
            'original_discovery_coverage':selection['coverage'],'recovery':recovered,
            'future_verification_status':'NOT_RUN','visual_evidence_in_original_discovery':False,
            'source_code_sha256':sha256(Path(__file__))}
    output=Path(output) if output is not None else root/'artifacts/private/m5/recovery'
    write_json(output/'recovery.json',result,immutable=True)
    write_json(output/'recovered_discovery.json',recovered['discovery'],immutable=True)
    write_json(output/'recovery.seal.json',{'files':{p.name:sha256(p) for p in [output/'recovery.json',output/'recovered_discovery.json']},
                                         'parent_review_seal_sha256':result['parent']['review_seal_sha256']},immutable=True)
    return result
