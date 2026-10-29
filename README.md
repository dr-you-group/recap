# RECAP

RECAP is a statistical reviewer agent for reported results in medical manuscripts. It reads PDF/DOCX files, links selected claims to source evidence, checks explicit simple-percentage calculations, and produces a review in JSON and Markdown. Novelty, impact, and accept/reject recommendations are outside its scope.

## Getting started

### 1. Install

Use Python 3.12 and `uv`. From the repository root:

```bash
uv sync --frozen --no-dev
uv run --no-sync recap --help
```

Run RECAP from this source checkout. The current runtime loads companion scripts and local infrastructure records relative to the repository; a standalone wheel installation is not sufficient for live document review.

### 2. Configure serve_llm

**Live document review requires the separately maintained [serve-llm infrastructure](https://github.com/ts06068/serve-llm), checked out at `./serve-llm/`.** It is not included in this repository and is not the similarly named PyPI package. Prepare its model server, credentials, SSH access, and deployment records before using `recap review` or `recap ground`.

If it is not already present, clone it from the RECAP repository root:

```bash
git clone https://github.com/ts06068/serve-llm.git serve-llm
```

The current runtime also requires the matching local infrastructure state:

- `serve-llm/runpod/common.py`, `serve-llm/runpod/deployment.json`, and `serve-llm/.tools/runpodctl`.
- Private endpoint/Runpod configuration under `serve-llm/.secrets/`.
- `artifacts/private/m1/us_deployment.json` and `artifacts/private/m5/known_hosts`.
- `artifacts/m1/us_resource_ledger.json`, `model_binding.json`, and `server_observation.json`, plus `artifacts/r2/model_binding.json`.

These are deployment-specific prerequisites, not benchmark results to download. Provision them through the existing infrastructure setup; do not fabricate identity records. They remain local and ignored. **An API URL/key alone does not make the current document runtime portable to a new server.** The retained checks are tied to the recorded deployment and model binding.

Endpoint credentials can be supplied through `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`, or through `--credentials /secure/path/llm.env`. The default credential path is `serve-llm/.secrets/llm.env`. The infrastructure checks additionally use their own local credential files. Never commit these files or put secret values in command arguments.

`profiles/final.json` records the retained `google/gemma-4-31B-it` configuration: a 2,048-token thinking budget, an 8,192-token output budget, and a 240-second request limit. It must match the configured server. RECAP does not start or replace the server automatically.

### 3. Review a manuscript

Once the infrastructure is ready:

```bash
uv run --no-sync recap review input.pdf \
  --profile profiles/final.json --output runs/paper-001 --allow-live

uv run --no-sync recap review input.docx \
  --profile profiles/final.json --output runs/manuscript-001 --allow-live
```

Use a new output directory for every run. The file extension selects PDF or DOCX handling. The default workflow checks up to six candidates from up to 30,000 selected characters; `--max-checks` and `--max-chars` adjust those limits. `--provenance asset.json` optionally supplies source/version/license metadata. Without `--allow-live`, the CLI does not invoke the model.

Read `result.json` for structured results and `report.md` for the review. The output directory also records source selections, evidence references, original responses, hashes, and coverage limitations. The adjacent `<output>.runtime/` directory contains execution traces and budget state. All generated outputs remain ignored by Git.

For native text extraction without model inference:

```bash
uv run --no-sync recap extract input.pdf --output runs/extracted.json
```

Run `uv run --no-sync recap <command> --help` for additional options, including experimental case-scoped visual grounding with `ground` and offline recalculation with `recalculate-review`.

### Current limits

Delivery remains partial: a previous live DOCX review completed, and PDF discovery linked six candidates and four summary items to source text, but PDF verification timed out after 240 seconds. The subsequent PDF output-limit change passed CPU checks and has not completed a fresh live validation. Native-text extraction does not add OCR. Empty findings do not certify an error-free manuscript, and simple-percentage checks do not cover all statistical methods.

## Benchmark

### Dataset and download

Evaluation used [SciClaimEval Subtask 1](https://sciclaimeval.github.io/), which asks whether a supplied claim is Supported or Refuted by supplied evidence. The initial sample contained **100 cases / 50 complete pairs / 27 PeerJ papers**, selected from public dev v1.1 with seed `20261013`, individual CC BY 4.0 licensing, and the allowed context conditions.

The downloader used the Hugging Face dataset [alabnii/sciclaimeval-shared-task at revision `55ec80b427d9b4eaff8ed9b537a1ea6cb5d665d7`](https://huggingface.co/datasets/alabnii/sciclaimeval-shared-task/tree/55ec80b427d9b4eaff8ed9b537a1ea6cb5d665d7). It fetched `data/dev_task1_release.json` and the original PNG evidence referenced by each selected row over HTTPS, caching source URLs, versions, and SHA-256 hashes. It did not replace benchmark images with figures from the original papers.

To download and prepare the same initial sample:

```bash
uv run --no-sync recap prepare-benchmark --output data/sciclaimeval
```

The first run needs network access; add `--offline` only when the raw cache already exists. Preparation separates model inputs under `bundle/` from gold labels under `scoring/targets.jsonl`. **All benchmark data, labels, manifests, predictions, and results are ignored and are not distributed with this repository.**

### Metrics

The primary metric is **pair accuracy**: both cases in a pair must be correct. Secondary metrics are case accuracy and macro-F1 over Supported/Refuted. Prediction coverage and failures are reported separately. Predictions are frozen before the independent scorer reads gold labels; incomplete runs receive full-denominator diagnostics instead of an official complete-run score.

The bundled scorer is an unmodified copy of the [official evaluation script at revision `79bb4f73d78c65c043481b6799bdd0810fad3298`](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/79bb4f73d78c65c043481b6799bdd0810fad3298/evaluation/eval_script.py), credited upstream to Alab-NII/SciTabAlign. Its SHA-256 is `14d0854a2e8da2a10867634c9fab2ca8c7cccdd8cff5bd89e6b2b42f9655b375`. The existing source audit found no upstream license grant for redistribution; that status remains unresolved and is separate from dataset licensing.

### Recorded results

The historical H0 baseline used `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, on the same 100-case development sample. Both settings used the same claim/evidence workflow, original PNGs, an 8,192-token output budget, and a 300-second request limit.

| H0 configuration | Valid predictions | Generation attempts | Accuracy | Pair accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reasoning enabled | 100/100 | 102 | 86.00% (86/100) | 72.00% (36/50) | 85.99% |
| Reasoning disabled | 100/100 | 100 | 78.00% (78/100) | 60.00% (30/50) | 77.92% |

The enabled run required two transport retries. These are recorded development results, not a new execution or an untouched test evaluation. They measure supplied-claim verification by the historical H0 baseline, **not the final PDF/DOCX review workflow**.

Later development comparisons did not establish an accuracy gain: a complete six-case/three-pair comparison tied at 100% for H0 and the compact candidate; a separate four-case/two-pair comparison had only two valid responses per arm and no official score (full-denominator diagnostics: 1/4 correct cases and 0/2 correct pairs for both). These small subsets must not be pooled with the 100-case baseline. Whole-manuscript accuracy and superiority over strong frontier systems remain unestablished.
