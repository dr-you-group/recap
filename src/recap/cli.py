"""Explicit phases; no model or network activity on import or offline tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark import prepare_benchmark
from .boundary import EvidenceBundle
from .contracts import CapabilityProfile, CaseInput, GenerationOptions
from .extract import extract_file
from .io import canonical, read_jsonl, write_json, write_jsonl
from .pmc import collect_pmc
from .provider import ChatCompletionsAdapter, load_credentials
from .runtime import predict_case
from .scoring import freeze_predictions, score_files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="recap", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-benchmark")
    prep.add_argument("--output", type=Path, default=Path("data/sciclaimeval"))
    prep.add_argument("--offline", action="store_true")
    pmc = commands.add_parser("collect-pmc")
    pmc.add_argument("--output", type=Path, default=Path("data/pmc"))
    pmc.add_argument("--target", type=int, choices=(5, 30), default=5)
    pmc.add_argument("--offline", action="store_true")
    extract = commands.add_parser("extract")
    extract.add_argument("input", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--profile", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--credentials", type=Path, default=Path("serve-llm/.secrets/llm.env"))
    predict.add_argument("--allow-live", action="store_true")
    predict.add_argument("--max-tokens", type=int, default=512)
    predict.add_argument("--reasoning", choices=("enabled", "disabled"))
    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--targets", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review", help="Discover and check manuscript reporting with source-linked host calculations")
    review.add_argument("input", type=Path)
    review.add_argument("--condition", choices=("pdf_only", "docx_native", "xml_assisted"), help="Defaults to pdf_only/docx_native from the input suffix")
    review.add_argument("--profile", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--max-checks", type=int, default=6)
    review.add_argument("--max-chars", type=int, default=30000)
    review.add_argument("--source-index", type=Path)
    review.add_argument("--provenance", type=Path, help="Optional source/version/license asset metadata; never discovery or answers")
    review.add_argument("--control", type=Path)
    review.add_argument("--credentials", type=Path, default=Path("serve-llm/.secrets/llm.env"))
    review.add_argument("--allow-live", action="store_true")
    review.add_argument("--workflow", choices=("m4", "m5-keyed"), default="m5-keyed",
                        help="m4 preserves the original path; m5-keyed uses fresh discovery and M5 isolated/keyed verification")
    ground=commands.add_parser('ground',help='Experimental case-scoped original crop and fresh verdict; never automatic accuracy selection')
    ground.add_argument('bundle',type=Path)
    ground.add_argument('--case-id',required=True)
    ground.add_argument('--proposal',type=Path,help='Optional previously recorded location-only JSON; otherwise one fresh P')
    ground.add_argument('--profile',type=Path,required=True)
    ground.add_argument('--output',type=Path,required=True)
    ground.add_argument('--control',type=Path)
    ground.add_argument('--credentials',type=Path,default=Path('serve-llm/.secrets/llm.env'))
    ground.add_argument('--compare',action='store_true',help='One original-only and one crop-added verdict share the same P')
    ground.add_argument('--allow-live',action='store_true')
    recalculate = commands.add_parser("recalculate-review", help="Recompute host arithmetic on sealed model outputs without generation")
    recalculate.add_argument("input", type=Path)
    recalculate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-benchmark":
        result = prepare_benchmark(args.output, offline=args.offline)
    elif args.command == "collect-pmc":
        result = collect_pmc(args.output, target=args.target, offline=args.offline)
    elif args.command == "extract":
        result = extract_file(args.input)
        write_json(args.output, result, immutable=True)
    elif args.command == "predict":
        # This branch must not read credentials or instantiate a network client without opt-in.
        if not args.allow_live:
            result = {"status": "NOT_RUN", "live_inference": "NOT_RUN", "serving_verified": False,
                      "reason": "requires explicit --allow-live in a separately authorized run"}
            print(json.dumps(result, indent=2))
            return 0
        if args.output.exists() or args.output.with_name(args.output.name + ".seal.json").exists():
            raise FileExistsError("run output already exists")
        profile = CapabilityProfile.model_validate_json(args.profile.read_text())
        provider = ChatCompletionsAdapter(profile, load_credentials(args.credentials), allow_live=True)
        bundle = EvidenceBundle(args.bundle)
        cases = [CaseInput.model_validate(r) for r in read_jsonl(args.bundle / "cases.jsonl")]
        code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))).hexdigest()
        options = GenerationOptions(output_budget=args.max_tokens, reasoning=args.reasoning)
        predictions, records = [], []
        for case in cases:
            prediction, record = predict_case(case, bundle, provider, options, code_version=code_hash)
            predictions.append(prediction)
            records.append(record.model_dump())
        result = freeze_predictions(args.output, predictions, execution_kind="live")
        write_jsonl(args.output.with_name(args.output.name + ".runs.jsonl"), records, immutable=True)
    elif args.command == "recalculate-review":
        from .document_review import recalculate_review
        result = recalculate_review(args.input, args.output)
        print(json.dumps({"status": result["status"], "new_generations": 0, "output": str(args.output)}))
        return 0
    elif args.command == 'ground':
        from .visual_cli import ground_cli
        result=ground_cli(args);print(json.dumps(result,ensure_ascii=False))
        return 2 if result['status']=='PARTIAL' else 0
    elif args.command == "review":
        if args.condition is None:
            args.condition={'.pdf':'pdf_only','.docx':'docx_native'}.get(args.input.suffix.lower())
            if args.condition is None:raise ValueError('PDF/DOCX suffix or explicit condition required')
        if args.workflow == "m5-keyed":
            from .document_cli_m6 import review_cli
        else:
            from .m4_execution import review_cli
        result = review_cli(args)
        print(json.dumps({k: result[k] for k in ("status", "document_id", "coverage", "calculation_execution_count", "stop_reason") if k in result}, ensure_ascii=False))
        return 2 if result.get("status") == "PARTIAL" else 0
    else:
        result = score_files(args.predictions, args.targets)
        write_json(args.output, result, immutable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") != "PARTIAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
