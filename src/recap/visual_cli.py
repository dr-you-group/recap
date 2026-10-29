"""Experimental public visual grounding over existing case bundles and final requests."""
from pathlib import Path
import json
from .boundary import EvidenceBundle
from .contracts import CaseInput,Prediction
from .visual_grounding import partition_plan,crop_source_regions,proposal_request,final_request
from .h0 import prediction_from_response,digest
from .io import now,sha256,write_json
from .provider import ModelResponse
from .final_runtime import session_for
from .result_durability import safe_error


def ground_cli(args):
    if not args.allow_live:return {'status':'NOT_RUN','reason':'explicit live opt-in required'}
    output=Path(args.output)
    if output.exists():raise FileExistsError('new output directory required')
    from .m6_scoring import read_selected_targets
    # This ID-only reader also accepts label-free CaseInput JSONL; no scoring target enters this path.
    case=CaseInput.model_validate(read_selected_targets(args.bundle/'cases.jsonl',[args.case_id])[0])
    bundle=EvidenceBundle(args.bundle);output.mkdir(parents=True)
    result={'at':now(),'status':'PARTIAL','experimental':True,'input_id':case.input_id,'arms':{},'crop_delivered':False,'semantic_target_accuracy':'unverified','new_generations':0}
    try:
        with session_for(args.profile,output,control=args.control,credentials=args.credentials) as live:
            original=json.loads(args.proposal.read_text()) if args.proposal else {'regions':[]}
            derived=partition_plan(original,case)
            if not derived['plan']['regions']:
                request,_=proposal_request(case,bundle,live.options)
                response=live.generate(request,lambda text:partition_plan(text,case),'proposal');result['new_generations']+=1
                if response.status=='ok' and response.observed['execution_state']['evaluation_eligibility']=='eligible':
                    original=json.loads(response.text);derived=partition_plan(original,case)
            plan={'input_id':case.input_id,'plan':derived['plan'],'contract_version':derived['contract_version'],
                'semantic_target_accuracy':'unverified_exploratory_not_gold','original_proposal_sha256':sha256(args.proposal) if args.proposal else None}
            write_json(output/'plan.json',{'original':original,'partition':derived,'shared_plan':plan},immutable=True)
            crop=crop_source_regions(bundle,case,original,output/'crops')
            write_json(output/'input.json',{'case':case.model_dump(),'source_assets':[bundle.assets[e.asset_id].model_dump() for e in case.evidence],'profile_sha256':sha256(args.profile)},immutable=True)
            result['geometry_valid_regions']=derived['geometry_valid_count'];result['rejected_regions']=len(derived['rejected_regions']);result['crop_available']=crop['crop_available']
            for arm in (['C_plan','C_zoom'] if args.compare else ['C_zoom']):
                request,info=final_request(case,bundle,live.options,arm,plan,crop,output/'crops')
                def validator(raw):
                    if prediction_from_response(case,ModelResponse(raw,'stop','ok',{},{}),[]).status!='ok':raise ValueError('invalid final schema')
                response=live.generate(request,validator,arm);result['new_generations']+=1
                prediction=prediction_from_response(case,response,[])
                record={'prediction':prediction.model_dump(),'execution_state':response.observed['execution_state'],'input_metadata':info,
                    'brief_reason':json.loads(response.text)['brief_reason'] if response.status=='ok' else None}
                write_json(output/(arm+'.json'),record,immutable=True);result['arms'][arm]=record
                if arm=='C_zoom' and info['crop_delivered'] and response.status=='ok':result['crop_delivered']=True
            result['status']='COMPLETED_EXPERIMENTAL' if all(x['execution_state']['evaluation_eligibility']=='eligible' for x in result['arms'].values()) else 'PARTIAL'
    except (ValueError,RuntimeError,TimeoutError,OSError) as error:
        result['error']=safe_error(error,'visual_grounding')
    budget_path=(args.control.parent if args.control else output.with_name(output.name+'.runtime'))/'budget.json'
    if budget_path.exists():
        namespace=digest(str(output.resolve()))[:16]+':'
        result['new_generations']=sum(k.startswith(namespace) for k in json.loads(budget_path.read_text()).get('requests',{}))
    for arm in (['C_plan','C_zoom'] if args.compare else ['C_zoom']):
        if arm not in result['arms']:result['arms'][arm]={'prediction':Prediction(input_id=case.input_id,task_kind=case.task_kind,status='NOT_RUN').model_dump()}
    write_json(output/'result.json',result,immutable=True)
    (output/'report.md').write_text('# Experimental visual grounding\n\n'+json.dumps({k:v for k,v in result.items() if k!='arms'},ensure_ascii=False,indent=2)+'\n\nSemantic localization and accuracy are unverified. Full original is retained.\n')
    files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'seal.json',{'files':files,'sha256':digest(files)},immutable=True)
    return {k:result[k] for k in ['status','input_id','crop_delivered','new_generations']}
