"""M5 derived PDF verification and native table review; immutable source evidence."""
from __future__ import annotations
import json
from pathlib import Path
from .anchoring_m5 import anchor_quote, recover_discovery
from .document_review import (Discovery, Verification, discovery_request, schema_for,
    permitted_sources, source_map, execute_percentage, resolve_observation, render_report, VERIFY)
from .provider import Message, ModelRequest, ImagePart
from .io import write_json, sha256
from .h0 import digest
from .sources_m5 import select_dependency_sources, execute_table_percentage

VERIFY_M5 = VERIFY + '''\nUse all supplied native sources and rendered page images relevant to the selected candidates. Each rendered image has an explicit source ID in image order. Visual observations cite that image source ID and use model_observed/unverified; image content need not be an exact native-text quote. Do not assert host-verified visual accuracy. For a native DOCX table, copy event numerator, total denominator and reported percent from their own cell literals, not a synthesized sentence. Cite explicit header/caption or table context for population, timepoint, outcome and simple-proportion meaning; no imputed denominator. Numeric value excludes a literal percent sign but its quote preserves the actual cell. Host verifies same table/row and headers independently. If no applicable percentage is present, calculation=null. Preserve uncertainty and reported values; do not calculate the answer yourself. Return one review per candidate.'''


def verification_v2(text, sources, candidates, images=()):
    value = Verification.model_validate_json(text)
    lookup = source_map(sources)
    image_ids = {x['source_id'] for x in images}
    allowed = set(lookup) | image_ids
    expected = {c.candidate_id for c in candidates}
    if not {x.candidate_id for x in value.reviews} <= expected:
        raise ValueError('verification candidate outside selected boundary')
    accepted, rejected = [], []
    # Check security boundaries on every raw item before isolating malformed groups.
    for item in value.reviews:
        for obs in item.observations:
            for ref in obs.sources:
                if ref.source_id not in allowed:
                    raise ValueError('observation source outside allowed boundary')
                if ref.source_id in image_ids and obs.epistemic_status not in {'model_observed','unverified'}:
                    raise ValueError('visual observation cannot claim native textual or verified provenance')
        if item.calculation:
            for q in [item.calculation.numerator,item.calculation.denominator,item.calculation.reported_percent,item.calculation.context]:
                if q is not None and q.source_id not in lookup:
                    raise ValueError('calculation source outside native text boundary')
    for candidate in candidates:
        group=[r for r in value.reviews if r.candidate_id==candidate.candidate_id]
        if len(group)!=1:
            rejected.append({'candidate_id':candidate.candidate_id,'items':[r.model_dump() for r in group],
                             'anchors':[],'reason':'duplicate_candidate_reviews' if group else 'missing_candidate_review'})
            continue
        item=group[0]
        anchors=[]
        if item.calculation:
            req=item.calculation
            for q in [req.numerator,req.denominator,req.reported_percent,req.context]:
                if q is None: continue
                if q.source_id not in lookup:
                    raise ValueError('calculation source outside native text boundary')
                anchors.append(anchor_quote(q,lookup[q.source_id]))
        invalid=[x for x in anchors if x['status']!='accepted']
        record={'item':item.model_dump(),'anchors':anchors}
        if invalid:
            record['reason']='required_calculation_quote_unresolved';rejected.append(record)
        else: accepted.append(record)
    return {'accepted':accepted,'rejected':rejected,'unverified':value.unverified,
            'original_candidates':len(expected)}


def verification_request_v2(document_id,sources,candidates,options,images=()):
    if len(images)>2: raise ValueError('maximum two actual page images')
    image_sources=[];parts=[]
    for i,x in enumerate(images):
        data=Path(x['path']).read_bytes()
        if sha256(Path(x['path']))!=x['sha256'] or not x.get('actual_render'):
            raise ValueError('rendered image provenance changed')
        image_sources.append({'source_id':x['source_id'],'location':x['location'],
                              'source_modality':'pdf_page_render','image_index':i,'text':''})
        parts.append(ImagePart('image/png',data))
    payload={'sources':permitted_sources(sources),'rendered_images':image_sources,
             'selected_candidates':[c.model_dump() for c in candidates]}
    return ModelRequest(document_id+':verification',(
        Message('system',VERIFY_M5),Message('user',json.dumps(payload,ensure_ascii=False),tuple(parts))),
        options,schema_for(Verification,[*sources,*image_sources]))


def _linked_summary(summary,findings,candidates):
    by_id={c.candidate_id:c for c in candidates}
    conflicts=[x for x in findings if (x.get('calculation') or {}).get('result',{}).get('status')=='inconsistent']
    result=[]
    for item in summary:
        x=dict(item);sids={q['source_id'] for q in x['sources']}
        refs=[]
        for finding in conflicts:
            candidate=by_id[finding['candidate_id']]
            if sids & {q.source_id for q in candidate.sources} or x['field']=='main_result':
                refs.append({'candidate_id':finding['candidate_id'],'status':'unresolved_reported_vs_calculated_conflict',
                             'relationship':'shared_source' if sids & {q.source_id for q in candidate.sources} else 'document_main_result_warning'})
        x['finding_refs']=refs;x['reported_value_preserved']=True
        result.append(x)
    return result


def render_m5(result):
    report=render_report(result)
    for item in result.get('summary',[]):
        if item.get('finding_refs'):
            needle=f"- {item['field']}: {item['value']}"
            report=report.replace(needle,needle+' [UNRESOLVED: '+', '.join(r['candidate_id'] for r in item['finding_refs'])+']',1)
    if result.get('visual_records'):
        report+='\n## Rendered-page observations (model assertions, accuracy unverified)\n\n'
        for record in result['visual_records']:
            report+=json.dumps(record,ensure_ascii=False)+'\n\n'
    return report+'\nDiscovery lineage: '+json.dumps(result.get('lineage',{}),ensure_ascii=False)+'\n'


def _seal(output,result):
    write_json(output/'result.json',result,immutable=True)
    (output/'report.md').write_text(render_m5(result))
    files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'review.seal.json',{'files':files,'sha256':digest(files)},immutable=True)


def review_m5(index,selected,provider,options,output,*,max_checks=6,recovered=None,images=(),lineage=None,result_enricher=None):
    output=Path(output)
    if output.exists():raise FileExistsError('M5 review already exists; no resume or retry')
    if max_checks<1:raise ValueError('positive candidate cap required')
    if index['condition']=='pdf_only' and any(s['source_modality']=='jats_xml_text' for s in index['sources']):
        raise ValueError('XML forbidden in PDF-only evidence')
    native=source_map(index['sources'])
    for s in selected['sources']:
        original=native.get(s['source_id'])
        if original is None or any(s.get(k)!=original.get(k) for k in ['asset_id','source_modality','location']):
            raise ValueError('selected source outside native host boundary')
        if s['text']!=original['text']:
            excerpt=s.get('excerpt',{})
            if (excerpt.get('start_char')!=0 or excerpt.get('end_char')!=len(s['text'])
                or not excerpt.get('truncated') or not original['text'].startswith(s['text'])):
                raise ValueError('selected source differs from original native text')
    ids=set(native)
    for x in images:
        if x['parent_sha256']!=index['source_asset']['sha256']:raise ValueError('image from different original PDF')
        if (index['condition']!='pdf_only' or x['source_id'] in ids or x.get('source_modality')!='pdf_page_render'
            or x.get('media_type')!='image/png' or not x.get('actual_render')):
            raise ValueError('invalid actual PDF image source boundary')
        ids.add(x['source_id'])
    output.mkdir(parents=True)
    for name,value in [('source_index',index),('selection',selected),('images',list(images))]:
        write_json(output/(name+'.json'),value,immutable=True)
    result={'document_id':index['document_id'],'condition':index['condition'],'status':'PARTIAL',
            'summary':[],'findings':[],'stage_results':[],'calculation_execution_count':0,
            'local_document_path':index.get('local_path'), 'source_asset':index['source_asset'],
            'source_locations':{s['source_id']:s['location'] for s in [*index['sources'],*images]},
            'lineage':lineage or {'fresh_discovery':True,'replayed_discovery':False},
            'coverage':{'discovery_sources':selected['coverage'],'discovered_candidates':0,'processed_candidates':0,
                        'not_processed':[],'parse_failed':index.get('parse_failed',[]),'candidate_recall':'unknown',
                        'whole_document_certified':False,'OCR':'NOT_RUN','pdf_visual_review':'NOT_RUN' if images else 'not_applicable',
                        'indexed_native_sources':len(index['sources'])}}
    candidates=[]
    try:
        if recovered is None:
            request=discovery_request(index['document_id'],selected['sources'],options,max_checks)
            response=provider(request,lambda t:recover_discovery(t,selected['sources']),'discovery')
            result['stage_results'].append({'stage':'discovery','status':response.status,'finish_reason':response.finish_reason})
            if response.status!='ok':raise ValueError('discovery_'+response.status)
            recovered=recover_discovery(response.text,selected['sources'])
        write_json(output/'discovery_recovery.json',recovered,immutable=True)
        discovery=Discovery.model_validate(recovered['discovery'])
        write_json(output/'discovery.json',discovery.model_dump(),immutable=True)
        result['summary']=[s.model_dump() for s in discovery.summary]
        candidates=discovery.candidates[:max_checks]
        result['coverage'].update(discovered_candidates=len(discovery.candidates),
             discovery_original_counts=recovered.get('original_counts'),
             discovery_rejected=recovered.get('rejected'),discovery_ambiguous=recovered.get('ambiguous'),
             not_processed=[c.model_dump() for c in discovery.candidates[max_checks:]])
        if not candidates:raise ValueError('no_accepted_candidates')
        evidence=select_dependency_sources(index,candidates,max_chars=30000,max_sources=60)
        write_json(output/'verification_selection.json',evidence,immutable=True)
        request=verification_request_v2(index['document_id'],evidence['sources'],candidates,options,images)
        write_json(output/'verification_input.json',{'input_id':request.input_id,'prompt_sha256':digest([m.text for m in request.messages]),
            'image_sha256':[x['sha256'] for x in images],'input_source_ids':[s['source_id'] for s in evidence['sources']],
            'bundle_sha256':digest({'sources':evidence['sources'],'images':[x['sha256'] for x in images]})},immutable=True)
        response=provider(request,lambda t:verification_v2(t,evidence['sources'],candidates,images),'verification')
        result['stage_results'].append({'stage':'verification','status':response.status,'finish_reason':response.finish_reason})
        result['coverage'].update(rendered_pages=[x['page'] for x in images],
            image_bytes_transmitted=bool(images and response.observed.get('transmitted_at')),
            pdf_visual_review='TRANSMITTED_RESPONSE_INVALID' if images else 'not_applicable')
        if response.status!='ok':raise ValueError('verification_'+response.status)
        validated=verification_v2(response.text,evidence['sources'],candidates,images)
        write_json(output/'verification.json',json.loads(response.text),immutable=True)
        write_json(output/'verification_validation.json',validated,immutable=True)
        lookup=source_map([*evidence['sources'],*images])
        visual_count=0
        for item in validated['accepted']:
            from .document_review import ReviewItem
            r=ReviewItem.model_validate(item['item'])
            calc=None
            if r.calculation:
                calc=(execute_table_percentage(r.calculation,evidence['sources']) if index['condition']=='docx_native'
                      else execute_percentage(r.calculation,evidence['sources']))
                result['calculation_execution_count']+=bool(calc.get('executed'))
            observations=[]
            for obs in r.observations:
                resolved=resolve_observation(obs,lookup)
                is_visual=any(lookup[q.source_id]['source_modality']=='pdf_page_render' for q in obs.sources)
                resolved['content_accuracy']='unverified_model_observation' if is_visual else 'not_independently_verified'
                if is_visual:visual_count+=1
                observations.append(resolved)
            category=('확인 가능한 계산 불일치' if calc and calc['result']['status']=='inconsistent' else
                      '계산 일치(해당 조건 한정)' if calc and calc['result']['status']=='consistent' else
                      '근거 불충분' if r.status=='insufficient_evidence' else '확인 필요')
            result['findings'].append({'candidate_id':r.candidate_id,'category':category,'observations':observations,
                'model_interpretation':r.interpretation,'calculation':calc,'unverified':r.unverified,'quote_anchors':item['anchors']})
        result['summary']=_linked_summary(result['summary'],result['findings'],candidates)
        result['coverage'].update(processed_candidates=len(validated['accepted']),
            rejected_verifications=validated['rejected'],verification_sources=evidence['coverage'],
            unresolved=discovery.unverified+validated['unverified'],visual_observations=visual_count,
            rendered_pages=[x['page'] for x in images],image_bytes_transmitted=bool(images),
            pdf_visual_review='MODEL_OBSERVATIONS_UNVERIFIED' if visual_count else 'NO_VISUAL_OBSERVATIONS' if images else 'not_applicable')
        result['status']='COMPLETED_BOUNDED_REVIEW' if not validated['rejected'] else 'PARTIAL'
    except (ValueError,RuntimeError,TimeoutError) as error:
        result['stop_reason']=str(error)
    done={f['candidate_id'] for f in result['findings']}
    result['coverage']['not_processed'] += [c.model_dump() for c in candidates if c.candidate_id not in done]
    if result_enricher is not None:result_enricher(result)
    _seal(output,result)
    return result
