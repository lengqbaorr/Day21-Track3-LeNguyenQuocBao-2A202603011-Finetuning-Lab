"""Reproduce the side experiment from source Parquet and explicit LLM annotations.

Run with the repository's .venv/Scripts/python.exe. No labels are inferred by
keyword rules: annotations.jsonl contains the decisions made after reading.
Only data_custom is writable; data/*.jsonl is read to inherit the schema and
exclude contamination. See CUSTOM_DATASET.md for the annotation protocol.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "source"
DATASET = "tarudesu/ViOCD"
API = f"https://huggingface.co/api/datasets/{DATASET}/parquet"
EXPECTED_PARQUET_SHA256 = {
    "train": "393960a9fd262f6b92da5a6652f66bfd3af463c13fbc1cf7803388e3030bd794",
    "test": "fb8bac9c6f6a9db293b9f4dcf816b59f4c846ffa72d0e49ee6884183382c002d",
}
INTENTS = ("doi_tra", "van_chuyen", "hoan_tien", "san_pham_loi", "hoi_thong_tin")
URGENCIES = ("cao", "trung_binh", "thap")
SENTIMENTS = ("tieu_cuc", "trung_tinh", "tich_cuc")
LABEL_KEYS = ("intent", "urgency", "product", "sentiment")
PROTECTED = ("src", "notebooks", "scripts", "tests", "data", "results", "adapters")
# This lexicon only proposes candidates; the annotator chooses the exact span.
PRODUCTS = ("điện thoại", "điện_thoại", "smartphone", "iphone", "samsung", "oppo",
            "redmi", "vsmart", "realme", "nokia", "xiaomi", "pin", "tai nghe",
            "quần", "áo", "váy", "đầm", "giày", "dép", "túi", "balo", "son",
            "kem", "sữa rửa mặt", "nước tẩy trang", "tinh chất", "serum", "mascara",
            "phấn", "chai xịt", "nhãn_hiệu", "game", "ứng dụng", "aP", "app",
            "phần mềm", "zalo", "facebook", "shopee", "lazada", "tiki", "bluezone")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?84[ .-]?(?:\(0\)[ .-]?)?|0)(?:[ .-]?\d){8,10}(?!\w)")
ORDER_CONTEXT = re.compile(r"(?i)((?:mã\s*(?:đơn(?:\s*hàng)?|order)|(?:đơn\s*hàng|order)\s*(?:số|#|code))\s*[:#=-]?\s*)([A-Z0-9][A-Z0-9_-]*\d[A-Z0-9_-]*|\d{4,})")
ORDER_TOKEN = re.compile(r"(?i)\b(?:DH|VN|OD|ORD|ORDER|SO|SPX|SPE|LZD)[-_]?\d{4,}[A-Z0-9_-]*\b")


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): sha(p) for name in PROTECTED
            for p in sorted((ROOT / name).rglob("*")) if p.is_file()}


def normalize(text: str) -> str:
    """Whitespace only: do not rewrite spelling, punctuation, or accents."""
    return " ".join(text.split())


def mask(text: str) -> tuple[str, dict[str, int]]:
    text = normalize(text)
    counts = {}
    text, counts["email"] = EMAIL.subn("[EMAIL]", text)
    # Order context precedes phone matching so numeric order IDs retain their type.
    text, n1 = ORDER_CONTEXT.subn(lambda m: m[1] + "[ORDER_CODE]", text)
    text, n2 = ORDER_TOKEN.subn("[ORDER_CODE]", text)
    counts["order_code"] = n1 + n2
    text, counts["phone"] = PHONE.subn("[PHONE]", text)
    return text, counts


def canonical(text: str) -> str:
    text, _ = mask(text)
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def near(a: str, b: str) -> bool:
    """Exact canonical equality, char ratio >= .88, OR 3-token Jaccard >= .80.

    Character ratio applies to all lengths (including very short reviews).
    Shingles require at least 6 tokens on both sides. This is a defined lexical
    near-duplicate check, not a claim of semantic paraphrase detection.
    """
    if a == b:
        return True
    if not a or not b:
        return False
    if min(len(a), len(b)) / max(len(a), len(b)) >= .88 / (2 - .88):
        matcher = SequenceMatcher(None, a, b, autojunk=False)
        if matcher.quick_ratio() >= .88 and matcher.ratio() >= .88:
            return True
    aa, bb = a.split(), b.split()
    if min(len(aa), len(bb)) >= 6:
        sa = {tuple(aa[i:i+3]) for i in range(len(aa)-2)}
        sb = {tuple(bb[i:i+3]) for i in range(len(bb)-2)}
        if len(sa & sb) / len(sa | sb) >= .80:
            return True
    return False


def parquet_rows() -> dict[str, list[dict]]:
    for split, expected in EXPECTED_PARQUET_SHA256.items():
        if sha(SOURCE / f"{split}-0.parquet") != expected:
            raise ValueError(f"Source snapshot changed: {split}; annotations need re-review.")
    sys.path.insert(0, str(HERE / "_vendor"))
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("Install pyarrow into data_custom/_vendor with pip --target (see documentation).") from error
    return {split: pq.read_table(SOURCE / f"{split}-0.parquet").to_pylist()
            for split in ("train", "test")}


def download() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    if not (HERE / "protected_before.json").exists():
        dump(HERE / "protected_before.json", protected_hashes())
    request = urllib.request.Request(API, headers={"User-Agent": "ViOCD-research-side-experiment/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        manifest = json.load(response)
    dump(SOURCE / "parquet_api.json", manifest)
    downloaded = []
    for split in ("train", "test"):
        urls = manifest["default"][split]
        if len(urls) != 1:
            raise ValueError("Expected exactly one Parquet shard per split; inspect source before changing.")
        path = SOURCE / f"{split}-0.parquet"
        with urllib.request.urlopen(urls[0], timeout=60) as response:
            path.write_bytes(response.read())
        if sha(path) != EXPECTED_PARQUET_SHA256[split]:
            raise ValueError(f"Upstream {split} snapshot changed; stop and re-review the annotations.")
        downloaded.append({"split": split, "url": urls[0], "file": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    for name, url in (("README.md", f"https://huggingface.co/datasets/{DATASET}/raw/main/README.md"),
                      ("dataset_api.json", f"https://huggingface.co/api/datasets/{DATASET}")):
        with urllib.request.urlopen(url, timeout=60) as response:
            (SOURCE / name).write_bytes(response.read())
    dump(SOURCE / "download_manifest.json", downloaded)
    print(json.dumps(downloaded, ensure_ascii=False, indent=2))


def candidates() -> None:
    raw = parquet_rows()
    counts = {}
    for split, rows in raw.items():
        proposed = []
        for index, row in enumerate(rows):
            text, masked = mask(row["review"])
            found = [word for word in PRODUCTS if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I)]
            if found:
                proposed.append({"split": split, "row_index": index, "source_id": row["Unnamed: 0"],
                                 "human_complaint": int(row["label"]), "domain": row["domain"],
                                 "input": text, "product_hints": found, "masked": masked})
        write_jsonl(HERE / f"candidates_{split}.jsonl", proposed)
        counts[split] = {"source_rows": len(rows), "product_candidates": len(proposed),
                         "no_product_hint": len(rows)-len(proposed),
                         "domain": dict(Counter(r["domain"] for r in rows)),
                         "complaint": dict(Counter(int(r["label"]) for r in rows))}
    dump(HERE / "candidate_counts.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def build() -> None:
    raw = parquet_rows()
    template = jsonl(ROOT / "data" / "train_seed.jsonl")[0]
    if set(template) != {"instruction", "input", "label", "output"} or tuple(template["label"]) != LABEL_KEYS:
        raise ValueError("Unexpected original schema")
    originals = [(p.name, i, canonical(r["input"])) for p in sorted((ROOT / "data").glob("*.jsonl"))
                 for i, r in enumerate(jsonl(p))]
    annotations = []
    with (HERE / "manual_annotations.tsv").open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            annotation = {"split": record["split"], "row_index": int(record["row_index"]),
                          "decision": "skip" if record["intent"] == "SKIP" else "select"}
            if annotation["decision"] == "skip":
                annotation.update(reason=record["rule"], rationale=record["rationale"])
            else:
                annotation.update(label={key: record[key] for key in LABEL_KEYS},
                                  sentiment_rule=record["rule"], rationale=record["rationale"])
            annotations.append(annotation)
    write_jsonl(HERE / "annotations.jsonl", annotations)
    reviewed = set()
    selected = {"train": [], "test": []}
    provenance = []
    rejected = []
    accepted_texts = []
    masks = Counter()
    for annotation in annotations:
        split, index = annotation["split"], annotation["row_index"]
        if split not in selected or (split, index) in reviewed:
            raise ValueError(f"Invalid or repeated source row: {split}/{index}")
        reviewed.add((split, index))
        source = raw[split][index]
        text, mask_counts = mask(source["review"])
        if annotation["decision"] == "skip":
            rejected.append({"split": split, "row_index": index, "reason": annotation["reason"]})
            continue
        label = annotation["label"]
        if tuple(label) != LABEL_KEYS or label["intent"] not in INTENTS or label["urgency"] not in URGENCIES or label["sentiment"] not in SENTIMENTS:
            raise ValueError(f"Invalid label: {split}/{index}")
        if not label["product"] or label["product"] not in text or label["product"] not in normalize(source["review"]):
            raise ValueError(f"Product is not a verbatim original span: {split}/{index}")
        human = int(source["label"])
        baseline = "tieu_cuc" if human == 1 else "trung_tinh"
        if label["sentiment"] != baseline and annotation.get("sentiment_rule") not in {"S2", "S3", "S4"}:
            raise ValueError(f"Missing sentiment override rule: {split}/{index}")
        norm = canonical(text)
        contaminated = [(name, i) for name, i, other in originals if near(norm, other)]
        duplicate = [(s, i) for s, i, other in accepted_texts if near(norm, other)]
        if contaminated or duplicate:
            rejected.append({"split": split, "row_index": index,
                             "reason": "original_corpus_overlap" if contaminated else "custom_duplicate",
                             "matches": contaminated or duplicate,
                             "duplicate_kind": "exact" if any(norm == other for _, _, other in originals + accepted_texts) else "near"})
            continue
        if len(selected[split]) >= (250 if split == "train" else 50):
            rejected.append({"split": split, "row_index": index, "reason": "quota_full"})
            continue
        row = {key: {"instruction": template["instruction"], "input": text, "label": label,
                     "output": json.dumps(label, ensure_ascii=False)}[key] for key in template}
        selected[split].append(row)
        accepted_texts.append((split, index, norm))
        masks.update(mask_counts)
        provenance.append({"file": "train_seed.jsonl" if split == "train" else "eval_target.jsonl",
                           "output_row_1based": len(selected[split]), "split": split, "row_index": index,
                           "source_id": source["Unnamed: 0"], "human_complaint": human, "domain": source["domain"],
                           "source_review_sha256": hashlib.sha256(source["review"].encode()).hexdigest(),
                           "masked": mask_counts, "sentiment_rule": annotation.get("sentiment_rule", "S1"),
                           "rationale": annotation["rationale"]})
    for split, expected in (("train", 250), ("test", 50)):
        if len(selected[split]) != expected:
            dump(HERE / "rejections.json", rejected)
            raise ValueError(f"{split}: got {len(selected[split])}, expected {expected}; see rejections.json")
    write_jsonl(HERE / "train_seed.jsonl", selected["train"])
    write_jsonl(HERE / "eval_target.jsonl", selected["test"])
    write_jsonl(HERE / "provenance.jsonl", provenance)
    dump(HERE / "rejections.json", rejected)
    statistics = {"source_counts": {s: len(rows) for s, rows in raw.items()},
                  "reviewed": len(reviewed), "selected": {s: len(rows) for s, rows in selected.items()},
                  "skipped": dict(Counter(r["reason"] for r in rejected)), "mask_counts": dict(masks),
                  "distributions": {s: {key: dict(Counter(r["label"][key] for r in rows))
                                         for key in LABEL_KEYS} for s, rows in selected.items()},
                  "domains": {s: dict(Counter(p["domain"] for p in provenance if p["split"] == s)) for s in selected},
                  "human_complaint": {s: dict(Counter(p["human_complaint"] for p in provenance if p["split"] == s)) for s in selected},
                  "sentiment_rules": dict(Counter(p["sentiment_rule"] for p in provenance)),
                  "protected_files_unchanged": protected_hashes() == json.loads((HERE / "protected_before.json").read_text(encoding="utf-8"))}
    dump(HERE / "build_report.json", statistics)
    verify()
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


def verify() -> None:
    raw = parquet_rows()
    template = jsonl(ROOT / "data" / "train_seed.jsonl")[0]
    provenance = jsonl(HERE / "provenance.jsonl")
    by_file = defaultdict(list)
    for p in provenance:
        by_file[p["file"]].append(p)
    customs = []
    for name, split, size in (("train_seed.jsonl", "train", 250), ("eval_target.jsonl", "test", 50)):
        rows = jsonl(HERE / name)
        if len(rows) != size or len(by_file[name]) != size:
            raise AssertionError(f"Wrong row count: {name}")
        for ordinal, (row, p) in enumerate(zip(rows, by_file[name]), 1):
            source = raw[split][p["row_index"]]
            assert p["split"] == split and p["output_row_1based"] == ordinal
            assert p["source_id"] == source["Unnamed: 0"]
            assert p["source_review_sha256"] == hashlib.sha256(source["review"].encode()).hexdigest()
            assert tuple(row) == tuple(template) and row["instruction"] == template["instruction"]
            assert tuple(row["label"]) == LABEL_KEYS and row["output"] == json.dumps(row["label"], ensure_ascii=False)
            assert row["input"] == mask(source["review"])[0]
            assert row["label"]["product"] in row["input"]
            assert row["label"]["intent"] in INTENTS and row["label"]["urgency"] in URGENCIES and row["label"]["sentiment"] in SENTIMENTS
            assert not EMAIL.search(row["input"]) and not PHONE.search(row["input"]) and not ORDER_TOKEN.search(row["input"])
            customs.append((name, ordinal, canonical(row["input"])))
    within_exact = within_near = cross_exact = cross_near = 0
    for i, (name, _, text) in enumerate(customs):
        for other_name, _, other in customs[:i]:
            if text == other:
                if name == other_name: within_exact += 1
                else: cross_exact += 1
            elif near(text, other):
                if name == other_name: within_near += 1
                else: cross_near += 1
    comparison = {}
    for path in sorted((ROOT / "data").glob("*.jsonl")):
        originals = [canonical(row["input"]) for row in jsonl(path)]
        exact = sum(text == other for _, _, text in customs for other in originals)
        approx = sum(text != other and near(text, other) for _, _, text in customs for other in originals)
        comparison[path.name] = {"rows": len(originals), "exact_overlaps": exact, "near_overlaps": approx}
    report = {"custom_rows": len(customs), "within_split_exact": within_exact, "within_split_near": within_near,
              "train_eval_exact": cross_exact, "train_eval_near": cross_near, "original_corpus": comparison,
              "regression_explicitly_checked": "eval_regression.jsonl" in comparison,
              "protected_files_unchanged": protected_hashes() == json.loads((HERE / "protected_before.json").read_text(encoding="utf-8"))}
    dump(HERE / "verification_report.json", report)
    assert not any((within_exact, within_near, cross_exact, cross_near))
    assert all(not r["exact_overlaps"] and not r["near_overlaps"] for r in comparison.values())
    assert report["regression_explicitly_checked"] and report["protected_files_unchanged"]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("download", "candidates", "build", "verify"))
    args = parser.parse_args()
    {"download": download, "candidates": candidates, "build": build, "verify": verify}[args.mode]()
