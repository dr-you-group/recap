"""Bounded PubMed discovery and current PMC Cloud article-version collection."""
from __future__ import annotations

import json
import hashlib
import re
import random
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlencode, urlsplit, unquote, quote
from lxml import etree

from .contracts import Asset
from .download import DownloadCache, xml_root
from .extract import extract_jats, extract_pdf, text
from .io import now, sha256, write_json, write_jsonl

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
SEED = 20261013
QUERY = ('pubmed pmc[filter] AND english[lang] AND humans[mh] '
         'AND ("2022/01/01"[dp] : "2025/12/31"[dp]) '
         'AND (randomized controlled trial[pt] OR observational study[pt] '
         'OR cohort[tiab] OR case-control[tiab] OR cross-sectional[tiab]) '
         'NOT (review[pt] OR meta-analysis[pt] OR editorial[pt] OR letter[pt])')
MAX_CANDIDATES = 200
MAX_BYTES = 5 * 1024**3


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value).lower())


def yes(value) -> bool:
    return value is True or str(value).lower() in {"yes", "true", "1"}


def no(value) -> bool:
    return value is False or str(value).lower() in {"no", "false", "0"}


def has_allowed_article_license(xml_data: bytes) -> bool:
    licenses = xml_root(xml_data).xpath("./front/article-meta/permissions/license")
    # URLs can be license attributes or child ext-links. A reference or figure's
    # CC notice must never be mistaken for the article's own permissions.
    return any(re.search(r"creativecommons\.org/(?:licenses/by/|publicdomain/zero/)",
                         etree.tostring(node, encoding="unicode")) for node in licenses)


def allowed_license(code: str) -> bool:
    return bool(re.fullmatch(r"CC BY(?: [1-4]\.0)?|CC0(?: 1\.0)?", code.upper().strip()))


def s3_location(value: str) -> tuple[str, str, str | None]:
    parsed = urlsplit(value)
    if parsed.scheme != "s3" or parsed.netloc != "pmc-oa-opendata":
        raise ValueError("unexpected PMC object origin")
    key = unquote(parsed.path.lstrip("/"))
    if ".." in PurePosixPath(key).parts:
        raise ValueError("unsafe object key")
    return S3 + quote(key, safe="/"), key, parse_qs(parsed.query).get("md5", [None])[0]


def parse_pubmed(data: bytes) -> list[dict]:
    root = xml_root(data)
    records = []
    for node in root.findall("PubmedArticle"):
        article = node.find("./MedlineCitation/Article")
        ids = {x.get("IdType"): text(x) for x in node.findall("./PubmedData/ArticleIdList/ArticleId")}
        title = text(article.find("ArticleTitle"))
        abstract = " ".join(text(x) for x in article.findall("./Abstract/AbstractText"))
        pts = [text(x) for x in article.findall("./PublicationTypeList/PublicationType")]
        combined = (title + " " + abstract).lower()
        if "Randomized Controlled Trial" in pts:
            design = "randomized_controlled_trial"
        elif re.search(r"case[ -]control", combined):
            design = "case_control"
        elif re.search(r"cross[ -]sectional", combined):
            design = "cross_sectional"
        elif "cohort" in combined:
            design = "cohort"
        else:
            design = "observational"
        records.append({"pmid": text(node.find("./MedlineCitation/PMID")), "pmcid": ids.get("pmc"),
                        "doi": ids.get("doi"), "title": title, "publication_types": pts,
                        "language": [text(x) for x in article.findall("Language")],
                        "design_stratum": design, "design_basis": "PubMed publication type then title/abstract keywords",
                        "protocol_title": bool(re.search(r"\bprotocol\b", title, re.I))})
    return records


def discover(cache: DownloadCache, destination: Path) -> list[dict]:
    params = {"db": "pubmed", "term": QUERY, "sort": "pub_date", "retmax": MAX_CANDIDATES, "retmode": "json"}
    query_key = hashlib.sha256(QUERY.encode()).hexdigest()[:16]
    search, search_info = cache.fetch(EUTILS + "esearch.fcgi?" + urlencode(params), f"pubmed/search_{query_key}.json", kind="json")
    response = json.loads(search.read_bytes())["esearchresult"]
    if response.get("errorlist"):
        raise ValueError("PubMed query was not accepted")
    ids = response["idlist"]
    if len(ids) > MAX_CANDIDATES:
        raise ValueError("candidate cap exceeded")
    records = []
    for start in range(0, len(ids), 50):
        url = EUTILS + "efetch.fcgi?" + urlencode({"db": "pubmed", "id": ",".join(ids[start:start+50]), "retmode": "xml"})
        path, _ = cache.fetch(url, f"pubmed/batch_{start:03d}.xml", kind="xml")
        records.extend(parse_pubmed(path.read_bytes()))
    if len({r["pmid"] for r in records}) != len(ids):
        raise ValueError("PubMed fetch did not return every candidate exactly once")
    groups = defaultdict(list)
    for record in records:
        groups[record["design_stratum"]].append(record)
    rng = random.Random(SEED)
    queues = {}
    for name in sorted(groups):
        items = sorted(groups[name], key=lambda r: r["pmid"])
        rng.shuffle(items)
        queues[name] = deque(items)
    ordered = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                ordered.append(queues[name].popleft())
    write_json(destination / "query.json", {"query": QUERY, "translation": response.get("querytranslation"),
               "queried_at": search_info["retrieved_at"], "sort": "pub_date", "seed": SEED,
               "candidate_cap": MAX_CANDIDATES, "returned_pmids": ids, "search_sha256": search_info["sha256"]}, immutable=True)
    write_jsonl(destination / "candidates.jsonl", ordered, immutable=True)
    return ordered


def versions(cache: DownloadCache, pmcid: str) -> list[tuple[dict, dict, Path]]:
    params = {"list-type": "2", "prefix": pmcid + ".", "delimiter": "/"}
    listing, _ = cache.fetch(S3 + "?" + urlencode(params), f"metadata/{pmcid}.listing.xml", kind="xml")
    root = xml_root(listing.read_bytes())
    if root.findtext("{*}IsTruncated") == "true":
        raise ValueError("unexpected truncated version listing; no implicit version selection")
    prefixes = [x.text for x in root.findall("{*}CommonPrefixes/{*}Prefix")]
    results = []
    for prefix in sorted(prefixes):
        version_name = prefix.rstrip("/")
        if not re.fullmatch(re.escape(pmcid) + r"\.\d+", version_name):
            continue
        path, info = cache.fetch(S3 + prefix + version_name + ".json", "metadata/" + version_name + ".json", kind="json")
        results.append((json.loads(path.read_bytes()), info, path))
    return results


def choose_version(records: list[tuple[dict, dict, Path]], bibliographic: dict):
    eligible = []
    reasons = []
    for metadata, info, path in records:
        reason = None
        if metadata.get("pmcid") != bibliographic["pmcid"] or str(metadata.get("pmid")) != bibliographic["pmid"]:
            reason = "identifier_mismatch"
        elif not yes(metadata.get("is_pmc_openaccess")) or not no(metadata.get("is_manuscript")):
            reason = "not_open_publisher_version"
        elif yes(metadata.get("is_retracted")):
            reason = "retracted"
        elif not allowed_license(metadata.get("license_code", "")):
            reason = "license_not_cc_by_or_cc0"
        elif not metadata.get("pdf_url") or not metadata.get("xml_url"):
            reason = "pdf_or_xml_unavailable"
        elif not bibliographic.get("doi") or normalize(metadata.get("doi", "")) != normalize(bibliographic["doi"]):
            reason = "doi_unverified"
        if reason:
            reasons.append({"version": metadata.get("version"), "reason": reason})
        else:
            eligible.append((metadata, info, path))
    if len(eligible) != 1:
        return None, {"reason": "ambiguous_publisher_versions" if len(eligible) > 1 else "no_eligible_version", "versions": reasons}
    return eligible[0], {"other_versions": reasons}


def verify_correspondence(metadata: dict, jats: dict, pdf: dict) -> dict:
    all_text = "\n".join(p["text"] for p in pdf["pages"])
    front_text = "\n".join(p["text"] for p in pdf["pages"][:3])
    pdf_norm = normalize(all_text)
    doi = normalize(metadata["doi"])
    jats_doi = normalize(jats["identifiers"].get("doi", ""))
    title_words = [normalize(w) for w in metadata["title"].split() if len(normalize(w)) > 2]
    front_norm = normalize(front_text)
    title_ratio = sum(w in front_norm for w in title_words) / max(1, len(title_words))
    label_checks = [{"id": item["id"], "label": item["label"],
                     "seen_in_pdf_text": normalize(item["label"]) in pdf_norm if item["label"] else False}
                    for item in jats["tables"] + jats["figures"]]
    # JATS often labels graphics "Fig." whereas the PDF says "Figure".
    for item in label_checks:
        alt = normalize(item["label"]).replace("figure", "fig")
        item["seen_in_pdf_text"] |= bool(alt and alt in pdf_norm.replace("figure", "fig"))
    checks = {"doi_matches_metadata_jats_pdf": bool(doi and doi == jats_doi and doi in pdf_norm),
              "title_matches_jats": normalize(metadata["title"]) == normalize(jats["title"]),
              "pdf_title_token_coverage": title_ratio,
              "pdf_has_native_text": any(p["text"].strip() for p in pdf["pages"]),
              "table_figure_label_checks": label_checks}
    verified = checks["doi_matches_metadata_jats_pdf"] and checks["title_matches_jats"] and title_ratio >= .9 and checks["pdf_has_native_text"]
    verified = verified and bool(label_checks) and all(x["seen_in_pdf_text"] for x in label_checks)
    return {"status": "verified_identifier_title_reference_correspondence" if verified else "version_unverified",
            "checks": checks, "content_equivalence": "not_exhaustively_verified",
            "cell_or_bbox_mapping": "unverified"}


def collect_article(cache: DownloadCache, destination: Path, record: dict) -> dict:
    if not record["pmcid"]:
        return {**record, "status": "excluded", "reason": "missing_pmcid"}
    if record["protocol_title"]:
        return {**record, "status": "excluded", "reason": "protocol_not_results_article"}
    chosen, version_notes = choose_version(versions(cache, record["pmcid"]), record)
    if chosen is None:
        return {**record, "status": "excluded", **version_notes}
    metadata, metadata_info, metadata_path = chosen
    prefix = f"{record['pmcid']}.{metadata['version']}"
    downloaded = []
    def retrieve(s3_url, kind, role, license_evidence):
        url, key, md5 = s3_location(s3_url)
        if not key.startswith(prefix + "/"):
            raise ValueError("metadata asset outside the selected version")
        if not md5:
            raise ValueError("source MD5 absent")
        path, info = cache.fetch(url, "articles/" + key, kind=kind, expected_md5=md5)
        media = {"pdf": "application/pdf", "xml": "application/xml", "image": info.get("content_type") or "image/unknown"}[kind]
        asset = Asset(asset_id=prefix + ":" + PurePosixPath(key).name, source_url=url, object_key=key,
            source_identity={"pmid": record["pmid"], "pmcid": record["pmcid"], "doi": record["doi"]},
            version=str(metadata["version"]), path=str(path.relative_to(destination)), media_type=media,
            sha256=info["sha256"], size=info["size"], retrieved_at=info["retrieved_at"], source_md5=md5,
            etag=info["etag"], last_modified=info["last_modified"], license=metadata["license_code"],
            license_evidence=license_evidence, rights_status="verified", work_family="doi:" + record["doi"].lower())
        downloaded.append({**asset.model_dump(), "role": role})
        return path
    xml_path = retrieve(metadata["xml_url"], "xml", "jats", "article-version metadata; JATS permissions verified below")
    jats = extract_jats(xml_path)
    license_evidence = json.dumps(jats["licenses"], ensure_ascii=False)
    if not has_allowed_article_license(xml_path.read_bytes()):
        return {**record, "status": "excluded", "reason": "jats_license_unverified", "assets": downloaded}
    if jats["article_type"] not in {"research-article", "brief-report"}:
        return {**record, "status": "excluded", "reason": "not_original_research_article", "assets": downloaded}
    pdf_path = retrieve(metadata["pdf_url"], "pdf", "pdf", license_evidence)
    pdf = extract_pdf(pdf_path)
    correspondence = verify_correspondence(metadata, jats, pdf)
    extracted = destination / "extracted" / prefix
    write_json(extracted / "jats.json", jats, immutable=True)
    write_json(extracted / "pdf.json", pdf, immutable=True)
    if correspondence["status"] == "version_unverified":
        return {**record, "status": "excluded", "reason": "version_unverified", "correspondence": correspondence, "assets": downloaded}
    media = {PurePosixPath(s3_location(u)[1]).name: u for u in metadata.get("media_urls", [])}
    figure_outcomes, used_media = [], set()
    credit_pattern = r"reproduced|reprinted|permission|copyright|©|adapted from|courtesy|third.party"
    for figure in jats["figures"]:
        if figure["rights_notices"] or re.search(credit_pattern, figure["caption"], re.I):
            figure_outcomes.append({"figure_id": figure["id"], "status": "rights_unresolved", "reason": "figure_specific_notice_or_credit"})
            continue
        for href in figure["hrefs"]:
            stem = PurePosixPath(href or "").stem
            candidates = [name for name in media if name == href or PurePosixPath(name).stem == stem]
            if len(candidates) != 1:
                figure_outcomes.append({"figure_id": figure["id"], "status": "unavailable", "href": href})
                continue
            name = candidates[0]
            if name not in used_media:
                retrieve(media[name], "image", "figure", license_evidence + "; no figure-specific override found")
                used_media.add(name)
            figure_outcomes.append({"figure_id": figure["id"], "status": "available", "file": name})
    return {**record, "status": "available", "article_version": metadata["version"],
            "license": metadata["license_code"], "license_evidence": jats["licenses"],
            "metadata_sha256": metadata_info["sha256"], "metadata_path": str(metadata_path.relative_to(destination)),
            "work_family": "doi:" + record["doi"].lower(), "work_family_status": "DOI_link_only_other_versions_unverified",
            "review_gold": False, "correspondence": correspondence, "assets": downloaded,
            "figure_outcomes": figure_outcomes, "supplementary_policy": "excluded_by_default",
            "not_downloaded_media": sorted(set(media) - used_media),
            "extracted": str(extracted.relative_to(destination)), "table_count": len(jats["tables"]),
            "figure_count": len(jats["figures"]), "pdf_pages": len(pdf["pages"]), **version_notes}


def collect_pmc(destination: Path, *, target: int = 5, offline: bool = False) -> dict:
    if target not in {5, 30}:
        raise ValueError("M0 only permits preflight=5 or target=30")
    destination = destination.resolve()
    cache = DownloadCache(destination / "raw", offline=offline, max_bytes=MAX_BYTES)
    if target == 30:
        gate = json.loads((destination / "preflight.json").read_text())
        if not gate.get("passed") or gate.get("accepted") != 5:
            raise ValueError("five-paper preflight must pass before expansion")
        manifest = destination / "preflight_manifest.jsonl"
        if sha256(manifest) != gate["manifest_sha256"]:
            raise ValueError("preflight manifest changed")
        # Revalidate every accepted raw asset; an old flag alone cannot open the gate.
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            for asset in row["assets"]:
                if sha256(destination / asset["path"]) != asset["sha256"]:
                    raise ValueError("preflight asset changed")
    candidates = discover(cache, destination)
    outcomes, accepted = [], []
    for record in candidates:
        if len(accepted) == target:
            break
        try:
            result = collect_article(cache, destination, record)
        except OverflowError:
            outcomes.append({**record, "status": "failed", "reason": "5_GiB_cap"})
            break
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            result = {**record, "status": "failed", "reason": type(error).__name__, "detail": str(error)[:250]}
        outcomes.append(result)
        if result["status"] == "available":
            accepted.append(result)
        print(json.dumps({"pmcid": record["pmcid"], "status": result["status"],
                          "reason": result.get("reason"), "accepted": len(accepted), "target": target}), flush=True)
    prefix = "preflight" if target == 5 else "pilot"
    write_jsonl(destination / (prefix + "_manifest.jsonl"), accepted, immutable=True)
    write_jsonl(destination / (prefix + "_outcomes.jsonl"), outcomes, immutable=True)
    passed = len(accepted) == target
    summary = {"status": "DONE" if passed else "PARTIAL", "passed": passed,
               "target": target, "accepted": len(accepted), "attempted": len(outcomes),
               "candidate_count": len(candidates), "candidate_cap": MAX_CANDIDATES,
               "download_cap_bytes": MAX_BYTES,
               "assets": sum(len(r["assets"]) for r in accepted),
               "asset_bytes": sum(a["size"] for r in accepted for a in r["assets"]),
               "design_strata": dict(Counter(r["design_stratum"] for r in accepted)),
               "excluded_or_failed": dict(Counter(r.get("reason") for r in outcomes if r["status"] != "available")),
               "figures_unavailable_or_unresolved": sum(f["status"] != "available" for r in accepted for f in r["figure_outcomes"]),
               "manifest_sha256": sha256(destination / (prefix + "_manifest.jsonl")),
               "query_sha256": sha256(destination / "query.json"), "review_gold": False}
    write_json(destination / (prefix + ".json"), summary, immutable=True)
    return {**summary, "cache_hits_this_run": cache.hits, "downloads_this_run": cache.downloads}
