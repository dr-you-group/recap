"""Public opt-in connection to M5 document functions; no document-specific cases.

Fresh discovery is followed by candidate-linked rendering and keyed verification.
This connection is offline tested; it does not certify fresh document live success.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from .anchoring_m5 import recover_discovery
from .document_review import Discovery, discovery_request
from .document_sources import index_document, select_sources
from .h0 import digest
from .io import sha256, write_json
from .provider import ModelResponse
from .result_durability import safe_error
from .m4_execution import LiveSession, ROOT
from .pdf_verification_m5 import VERSION, BOUNDED, OUTPUT_LIMITS, make_keyed_request, validate_keyed
from .review_m5 import review_m5
from .sources_m5 import build_dependency_index, render_referenced_pages, select_dependency_sources


def run_keyed_document(index, selected, generate, options, output, *, render_dir, max_checks=6,execution_kind='mock_contract',discovery_mode='quotes',verification_mode='legacy'):
    """Reuse M5 validation/calculation, preserving raw outputs and one call per stage."""
    output, render_dir = Path(output), Path(render_dir)
    if output.exists():
        raise FileExistsError("review output exists; no resume or retry")
    if max_checks < 1:
        raise ValueError("positive candidate cap required")
    if verification_mode not in {'legacy',BOUNDED}:
        raise ValueError('unsupported verification mode')
    if verification_mode==BOUNDED and index['condition']!='pdf_only':
        raise ValueError('bounded verification is a PDF-only final contract')
    if index["condition"] == "pdf_only" and any(s["source_modality"] == "jats_xml_text" for s in index["sources"]):
        raise ValueError("XML forbidden in PDF-only evidence")
    native = {source["source_id"]: source for source in index["sources"]}
    for source in selected["sources"]:
        original = native.get(source["source_id"])
        if original is None or any(source.get(k) != original.get(k) for k in ("asset_id", "source_modality", "location")):
            raise ValueError("selected source outside native host boundary")
        if source["text"] != original["text"]:
            excerpt = source.get("excerpt", {})
            if (excerpt.get("start_char") != 0 or excerpt.get("end_char") != len(source["text"])
                    or not excerpt.get("truncated") or not original["text"].startswith(source["text"])):
                raise ValueError("selected source differs from native host text")
    catalog, discovery_projection = None, None
    if discovery_mode == 'source_spans_v1':
        from .discovery_spans import catalog_for, request_for, project_discovery
        catalog = catalog_for(selected['sources'])
        request = request_for(index['document_id'], catalog, options, max_checks) if catalog['references'] else None
        def decode_discovery(text):
            projection = project_discovery(text, catalog, selected['sources'], max_checks)
            return recover_discovery(projection['discovery'], selected['sources'])
    elif discovery_mode == 'quotes':
        request = discovery_request(index["document_id"], selected["sources"], options, max_checks)
        def decode_discovery(text):
            return recover_discovery(text, selected['sources'])
    else:
        raise ValueError('unsupported discovery mode')
    response = (generate(request, decode_discovery, "discovery") if request is not None else
        ModelResponse('', None, 'failed', {}, {'diagnostic':'NOT_RUN_no_native_source_text'}))
    captured, images, recovery, preparation_error = {}, [], None, None
    if response.status == "ok":
        try:
            recovery = decode_discovery(response.text)
            if catalog is not None:
                discovery_projection = project_discovery(response.text, catalog, selected['sources'], max_checks)
            candidates = Discovery.model_validate(recovery["discovery"]).candidates[:max_checks]
            if candidates and index["condition"] == "pdf_only":
                images = render_referenced_pages(index, candidates, render_dir, dpi=120, max_pages=2)["images"]
        except (ValueError, RuntimeError, OSError) as exc:
            preparation_error = type(exc).__name__ + ": " + str(exc)
            recovery = None
    called = set()

    def provider(current, validator, stage):
        if stage in called:
            raise RuntimeError("document stage already attempted; no retries")
        called.add(stage)
        if stage == "discovery":
            # This is the original response, never a second discovery generation.
            return response
        if stage != "verification":
            raise ValueError("unsupported document stage")
        if preparation_error:
            raise ValueError(preparation_error)
        candidates = Discovery.model_validate(recovery["discovery"]).candidates[:max_checks]
        evidence = select_dependency_sources(index, candidates, max_chars=30000, max_sources=60)
        keyed = make_keyed_request(current, candidates, images,output_contract=verification_mode)
        write_json(output / "keyed_verification_input.json", {
            "version": VERSION, "output_contract":verification_mode, "input_id": keyed.input_id, "schema": keyed.output_schema,
            "prompt_sha256": digest([m.text for m in keyed.messages]),
            "image_sha256": [image["sha256"] for image in images]}, immutable=True)

        def validate(raw):
            projected = validate_keyed(raw, candidates, images, sources=evidence["sources"],output_contract=verification_mode)
            validator(json.dumps(projected["verification"], ensure_ascii=False))
            return projected

        result = generate(keyed, validate, "verification")
        # Failure text is retained as-is; no projection/repair or extra generation.
        write_json(output / "raw_keyed_response.json", asdict(result), immutable=True)
        if result.status != "ok":
            return result
        projected = validate(result.text)
        captured.update(projected)
        write_json(output / "keyed_projection.json", projected, immutable=True)
        return replace(result, text=json.dumps(projected["verification"], ensure_ascii=False),
                       observed={**result.observed, "host_projection": projected["normalization_meta"]})

    def enrich(result):
        result['discovery_method'] = discovery_mode
        result['verification_contract'] = verification_mode
        if verification_mode==BOUNDED:
            result['coverage']['verification_output_limits']=dict(OUTPUT_LIMITS)
            result['coverage']['verification_detail_scope']='Bounded observations and narrative per candidate/page; all selected candidate keys required; numerical evidence contract unchanged. Not exhaustive evidence enumeration.'
            result['coverage']['calculations_not_requested']=[
                {'candidate_id':finding['candidate_id'],'status':'not_applicable','executed':False,
                 'scope':'No typed simple-percentage request supplied; applicability of other statistical checks remains unverified.'}
                for finding in result.get('findings',[]) if finding['calculation'] is None]
        if catalog is not None:
            write_json(output / 'discovery_catalog.json', catalog, immutable=True)
            if discovery_projection is not None:
                write_json(output / 'discovery_projection.json', discovery_projection, immutable=True)
            result['coverage']['discovery_reference_selection_accuracy'] = 'unverified_model_selection'
        write_json(output / "discovery_response.json", asdict(response), immutable=True)
        if recovery is not None:
            result["stage_results"].insert(0, {"stage": "discovery", "status": response.status,
                                               "finish_reason": response.finish_reason})
        result["public_workflow"] = "m5-keyed"
        completed=execution_kind=='live' and result['status']=='COMPLETED_BOUNDED_REVIEW' and response.status=='ok'
        result["zero_touch_document_live_verified"] = completed
        result['execution_kind']=execution_kind
        result["workflow_validation"] = 'fresh_cli_bounded_live_completed_generalization_unverified' if completed else 'bounded_workflow_incomplete_or_offline'
        if captured:
            records = captured["visual_records"]
            result["visual_records"] = records
            result["normalization_meta"] = captured["normalization_meta"]
            result["coverage"].update(visual_page_records=len(records),
                visual_page_observations=sum(len(r["observations"]) for r in records),
                visual_unreadable_pages=[r["page"] for r in records if r["unreadable"]])
            if images:
                result["coverage"].update(
                    pdf_visual_review="MODEL_OBSERVATIONS_UNVERIFIED" if any(r["observations"] for r in records) else "UNREADABLE",
                    visual_content_accuracy="unverified_model_observation")

    return review_m5(index, selected, provider, options, output, max_checks=max_checks,
        recovered=recovery, images=images,
        lineage={"fresh_discovery": True, "replayed_discovery": False,
                 "manual_corrective_generation": 0, "automatic_retries": 0,
                 "public_workflow": "m5-keyed"}, result_enricher=enrich)


def review_cli(args):
    if not args.allow_live:
        return {"status": "NOT_RUN", "reason": "explicit live opt-in required",
                "workflow": "m5-keyed", "zero_touch_document_live_verified": False}
    if args.output.exists():
        raise FileExistsError("review output exists; no resume or retry")
    previous = json.loads(args.source_index.read_text()) if args.source_index else None
    if previous and (previous["condition"] != args.condition or previous["source_asset"]["sha256"] != sha256(args.input)):
        raise ValueError("source index differs from input/condition")
    provenance=json.loads(args.provenance.read_text()) if getattr(args,'provenance',None) else None
    index = build_dependency_index(index_document(args.input, args.condition,
        provenance=previous["source_asset"] if previous else provenance))
    if previous:
        expected = build_dependency_index(previous)
        if index["sources"] != expected["sources"] or index["inventory"] != expected["inventory"]:
            raise ValueError("source index differs from actual native input")
    selected = select_sources(index, max_chars=args.max_chars, max_sources=60)
    spec=json.loads(args.profile.read_text())
    if spec.get('runtime')=='final':
        from .final_runtime import session_for
        from .contracts import GenerationOptions
        runtime=args.output.with_name(args.output.name+'.runtime')
        modes={'discovery_mode':spec.get('pdf_discovery','quotes') if index['condition']=='pdf_only' else 'quotes',
               'verification_mode':spec.get('pdf_verification','legacy') if index['condition']=='pdf_only' else 'legacy'}
        try:
            live=session_for(args.profile,args.output,control=args.control,credentials=args.credentials)
        except (ValueError,RuntimeError,TimeoutError,OSError) as error:
            detail=safe_error(error,'initial_precheck');write_json(runtime/'initial_failure.json',detail,immutable=True)
            def unavailable(*_):return ModelResponse('',None,'failed',{}, {'diagnostic':'NOT_RUN_initial_precheck','exception':detail})
            return run_keyed_document(index,selected,unavailable,GenerationOptions.model_validate(spec['options']),args.output,render_dir=runtime/'pages',max_checks=args.max_checks,execution_kind='NOT_RUN',**modes)
        with live:
            def generate(request,validator,stage):
                try:
                    response=live.generate(request,validator,stage)
                    if response.status=='ok' and response.observed['execution_state']['evaluation_eligibility']!='eligible':
                        return replace(response,text='',status='failed',observed={**response.observed,'diagnostic':'quarantined_final_preserved_in_raw'})
                    return response
                except (ValueError,RuntimeError,TimeoutError,OSError) as error:
                    detail=safe_error(error,stage);write_json(runtime/(stage+'.failure.json'),detail,immutable=True)
                    return ModelResponse('',None,'failed',{}, {'diagnostic':'stage_failed_or_NOT_RUN','exception':detail})
            return run_keyed_document(index,selected,generate,live.options,args.output,render_dir=runtime/'pages',max_checks=args.max_checks,execution_kind='live',**modes)
    identity = digest({"output": str(args.output.resolve())})[:16]
    attempts = ROOT / "artifacts/private/m6" / ("document_" + identity + ".attempts")
    with LiveSession(args.profile, attempts, allow_live=True, credentials_path=args.credentials,
                     control_path=args.control, milestone="m6") as live:
        return run_keyed_document(index, selected, live.generate, live.options, args.output,
                                  render_dir=attempts / "pages", max_checks=args.max_checks)
