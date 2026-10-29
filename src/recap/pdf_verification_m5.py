"""M5 PDF output-shape repair only: unique candidate/page keys, raw text unchanged.

This module performs no generation and never promotes visual content to verified
truth. Its common-contract projection must still pass normal source/quote checks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Literal

from pydantic import Field

from .contracts import Contract
from .document_review import Verification
from .provider import Message, ModelRequest

VERSION='M5_pdf_keyed_v2'
BOUNDED='bounded_v1'
OUTPUT_LIMITS={'candidate_observations':2,'observation_value_chars':160,
    'observation_id_chars':64,'interpretation_chars':180,
    'candidate_unverified':2,'visual_observations':2,'visual_limitations':2,
    'global_unverified':4,'uncertainty_chars':120}
KINDS=['layout','row_column','axis','unit','group_legend','value','direction','text','other']
INSTRUCTION='''\nOutput-shape requirement: reviews is one JSON object keyed by the exact supplied candidate IDs. Put all observations for a candidate within its single keyed review object; do not repeat candidate objects or emit candidate_id inside them. visual_pages is a separate JSON object keyed by every supplied rendered-image source ID. For each actual image, report up to six concise visible observations (kind and value), unreadable, and limitations. If no content can be read reliably, set unreadable=true and retain uncertainty rather than inventing observations. Do not infer visual observations from native text alone. Visual observations remain model assertions, not host-verified facts. Keep the same verification task, original evidence, native calculation requests and source boundaries. Return the specified JSON schema only.'''
BOUNDED_INSTRUCTION='''\nUse concise records within the attached schema limits. Keep every selected candidate and every supplied image key. For each candidate, state at most two distinct relevant facts with their sources; preserve both sides of a conflict using multiple source references where needed. Do not repeat manuscript paragraphs, the candidate description, or the same fact in the interpretation. State only the comparison or remaining uncertainty in the interpretation. Record at most two short visual observations per image; do not duplicate the native text review. If evidence is insufficient, say so; do not omit the candidate or invent missing evidence. Numeric evidence quotes and calculation applicability checks retain their original contract. These limits bound reported detail, not the scope of candidate coverage or proof of manuscript accuracy.'''


def _bounded(output_contract):
    if output_contract not in {'legacy',BOUNDED}:
        raise ValueError('unsupported keyed output contract')
    return output_contract==BOUNDED


def _limit_text(value,maximum):
    if len(value)>maximum:raise ValueError('bounded verification text exceeds limit')


def _limit_items(values,maximum,chars=None):
    if len(values)>maximum:raise ValueError('bounded verification collection exceeds limit')
    if chars is not None:
        for value in values:_limit_text(value,chars)


class VisualObservation(Contract):
    kind: Literal['layout','row_column','axis','unit','group_legend','value','direction','text','other']
    value: str = Field(min_length=1,max_length=500)


class VisualPage(Contract):
    observations: list[VisualObservation] = Field(max_length=6)
    unreadable: bool
    limitations: list[str] = Field(max_length=6)


def _candidate_ids(candidates):
    ids=[c.candidate_id if hasattr(c,'candidate_id') else c['candidate_id'] for c in candidates]
    if not ids or any(not isinstance(c,str) or not c for c in ids) or len(ids)!=len(set(ids)):
        raise ValueError('nonempty unique candidate IDs required')
    return ids


def _image_map(images):
    if len(images)>2:raise ValueError('maximum two actual PDF page images')
    result={}
    for image in images:
        sid=image.get('source_id')
        if (not isinstance(sid,str) or sid in result or not image.get('actual_render')
                or image.get('source_modality')!='pdf_page_render' or image.get('media_type')!='image/png'):
            raise ValueError('invalid or duplicate host rendered image provenance')
        result[sid]=image
    return result


def keyed_schema(original_schema,candidates,images,*,output_contract='legacy'):
    bounded=_bounded(output_contract)
    ids=_candidate_ids(candidates);image_map=_image_map(images)
    schema=deepcopy(original_schema)
    if 'ReviewItem' not in schema.get('$defs',{}):raise ValueError('expected common Verification schema')
    review=deepcopy(schema['$defs']['ReviewItem'])
    review['properties'].pop('candidate_id',None)
    review['required']=[x for x in review.get('required',[]) if x!='candidate_id']
    review['title']='KeyedReviewItem'
    schema['$defs']['KeyedReviewItem']=review
    visual=VisualPage.model_json_schema()
    for name,definition in visual.pop('$defs',{}).items():
        if name in schema['$defs']:raise ValueError('schema definition collision')
        schema['$defs'][name]=definition
    schema['$defs']['KeyedVisualPage']=visual
    schema['title']='KeyedPDFVerificationM5'
    schema['properties']={
        'reviews':{'type':'object','properties':{cid:{'$ref':'#/$defs/KeyedReviewItem'} for cid in ids},
                   'required':ids,'additionalProperties':False},
        'visual_pages':{'type':'object','properties':{sid:{'$ref':'#/$defs/KeyedVisualPage'} for sid in image_map},
                        'required':list(image_map),'additionalProperties':False},
        'unverified':deepcopy(original_schema['properties']['unverified'])}
    schema['required']=['reviews','visual_pages','unverified'];schema['additionalProperties']=False
    if bounded:
        limits=OUTPUT_LIMITS
        review['properties']['observations']['maxItems']=limits['candidate_observations']
        review['properties']['interpretation']['maxLength']=limits['interpretation_chars']
        obs=schema['$defs']['Observation']['properties']
        obs['value']['maxLength']=limits['observation_value_chars']
        obs['record_id']['maxLength']=limits['observation_id_chars']
        schema['$defs']['VisualObservation']['properties']['value']['maxLength']=limits['observation_value_chars']
        visual['properties']['observations']['maxItems']=limits['visual_observations']
        for field,count in [(review['properties']['unverified'],limits['candidate_unverified']),
                            (visual['properties']['limitations'],limits['visual_limitations']),
                            (schema['properties']['unverified'],limits['global_unverified'])]:
            field['maxItems']=count;field['items']['maxLength']=limits['uncertainty_chars']
    return schema


def make_keyed_request(old_request,candidates,images,*,output_contract='legacy'):
    bounded=_bounded(output_contract)
    image_map=_image_map(images)
    actual=[image for message in old_request.messages for image in message.images]
    if len(actual)!=len(image_map):raise ValueError('host image list differs from actual request image count')
    for part,host in zip(actual,image_map.values()):
        if part.media_type!='image/png' or hashlib.sha256(part.data).hexdigest()!=host['sha256']:
            raise ValueError('host image hash differs from actual request bytes/order')
    messages=list(old_request.messages)
    if not messages or messages[0].role!='system':raise ValueError('expected an existing system verification instruction')
    instruction=INSTRUCTION.replace('up to six concise','up to two concise') if bounded else INSTRUCTION
    first=messages[0];messages[0]=Message(first.role,first.text+instruction+(BOUNDED_INSTRUCTION if bounded else ''),first.images)
    suffix=':'+BOUNDED if bounded else ''
    return ModelRequest(old_request.input_id+':'+VERSION+suffix,tuple(messages),old_request.options,
                        keyed_schema(old_request.output_schema,candidates,images,output_contract=output_contract),old_request.tools)


def _strict_json(raw):
    if not isinstance(raw,str):raise ValueError('raw final JSON text required')
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key rejected at any depth')
            result[key]=value
        return result
    def bad_constant(value):raise ValueError('non-finite JSON constant prohibited')
    return json.loads(raw,object_pairs_hook=object_pairs,parse_constant=bad_constant)


def validate_keyed(raw,candidates,images,*,sources=None,output_contract='legacy'):
    bounded=_bounded(output_contract)
    value=_strict_json(raw);ids=_candidate_ids(candidates);image_map=_image_map(images)
    if not isinstance(value,dict) or set(value)!={'reviews','visual_pages','unverified'}:
        raise ValueError('keyed top-level schema mismatch')
    if not isinstance(value['reviews'],dict) or set(value['reviews'])!=set(ids):
        raise ValueError('keyed candidate set mismatch')
    if not isinstance(value['visual_pages'],dict) or set(value['visual_pages'])!=set(image_map):
        raise ValueError('keyed rendered page set mismatch')
    reviews=[]
    for cid in ids:
        item=value['reviews'][cid]
        if not isinstance(item,dict) or 'candidate_id' in item:raise ValueError('candidate ID belongs only to the host object key')
        reviews.append({'candidate_id':cid,**item})
    common=Verification.model_validate({'reviews':reviews,'unverified':value['unverified']},strict=True).model_dump()
    if bounded:
        limits=OUTPUT_LIMITS
        _limit_items(common['unverified'],limits['global_unverified'],limits['uncertainty_chars'])
        for review in common['reviews']:
            _limit_items(review['observations'],limits['candidate_observations'])
            _limit_text(review['interpretation'],limits['interpretation_chars'])
            _limit_items(review['unverified'],limits['candidate_unverified'],limits['uncertainty_chars'])
            for observation in review['observations']:
                _limit_text(observation['value'],limits['observation_value_chars'])
                _limit_text(observation['record_id'],limits['observation_id_chars'])
    if sources is not None:
        native={s['source_id'] for s in sources}
        if len(native)!=len(sources) or native & set(image_map):raise ValueError('native/image host source IDs collide')
        allowed=native|set(image_map)
        for review in common['reviews']:
            for obs in review['observations']:
                if any(ref['source_id'] not in allowed for ref in obs['sources']):raise ValueError('review source outside approved bundle')
            if review['calculation']:
                for field in ['numerator','denominator','reported_percent','context']:
                    q=review['calculation'][field]
                    if q is not None and q['source_id'] not in native:raise ValueError('calculation requires native source')
    visual_records=[]
    for sid,host in image_map.items():
        page=VisualPage.model_validate(value['visual_pages'][sid],strict=True).model_dump()
        if bounded:
            _limit_items(page['observations'],limits['visual_observations'])
            _limit_items(page['limitations'],limits['visual_limitations'],limits['uncertainty_chars'])
            for observation in page['observations']:_limit_text(observation['value'],limits['observation_value_chars'])
        if not page['unreadable'] and not page['observations']:
            raise ValueError('readable page requires at least one explicit model observation')
        if any(len(x)>500 for x in page['limitations']):raise ValueError('visual limitation exceeds bounded record')
        visual_records.append({'source_id':sid,'asset_id':host['asset_id'],
            'source_modality':'pdf_page_render','sha256':host['sha256'],'parent_sha256':host['parent_sha256'],
            'page':host['page'],'location':host['location'],'raw_model_record':page,
            'observations':[dict(obs,epistemic_status='unverified' if page['unreadable'] else 'model_observed') for obs in page['observations']],
            'unreadable':page['unreadable'],'limitations':page['limitations'],
            'host_provenance_verified':True,'visual_content_accuracy':'unverified_model_observation'})
    return {'verification':common,'visual_records':visual_records,'normalization_meta':{
        'version':VERSION,'output_contract':output_contract,
        'output_limits':dict(OUTPUT_LIMITS) if bounded else None,
        'raw_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'raw_text_modified':False,'candidate_ids_from_host_keys':ids,'image_source_ids_from_host_keys':list(image_map),
        'duplicate_keys_rejected':True,'projection':'Object keys expanded to existing common candidate_id fields; separate visual page records retain model values.',
        'semantic_values_modified':False,'source_quote_validation':'Must still pass existing native source/anchor validator',
        'visual_content_verified':False}}
