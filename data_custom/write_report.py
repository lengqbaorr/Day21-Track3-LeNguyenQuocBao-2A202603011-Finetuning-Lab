"""Render current distributions and the already completed manual review notes."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(name):
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines()]


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows]) + "\n"


report = json.loads((HERE / "build_report.json").read_text(encoding="utf-8"))
verification = json.loads((HERE / "verification_report.json").read_text(encoding="utf-8"))
distributions = report["distributions"]
sections = ["## Final counts and distributions\n", "| Output | Source split | Rows |\n| --- | --- | --- |\n| train_seed.jsonl | train | 250 |\n| eval_target.jsonl | test | 50 |\n"]
for key, values in (("intent", ("doi_tra", "van_chuyen", "hoan_tien", "san_pham_loi", "hoi_thong_tin")),
                    ("urgency", ("cao", "trung_binh", "thap")),
                    ("sentiment", ("tieu_cuc", "trung_tinh", "tich_cuc"))):
    sections.append(f"### {key}\n\n" + table(["Label", "Train", "Eval"],
                    [[value, distributions["train"][key].get(value, 0), distributions["test"][key].get(value, 0)] for value in values]))
sections.append("### Source domains\n\n" + table(["Domain", "Train", "Eval"],
                [[value, report["domains"]["train"].get(value, 0), report["domains"]["test"].get(value, 0)]
                 for value in ("mobile", "app", "fashion", "cosmetic")]))
products = sorted(set(distributions["train"]["product"]) | set(distributions["test"]["product"]))
sections.append("### Product spans (complete distribution)\n\n" + table(["Verbatim span", "Train", "Eval"],
                [[value, distributions["train"]["product"].get(value, 0), distributions["test"]["product"].get(value, 0)] for value in products]))
sections.append("### Original corpus decontamination\n\n" + table(["Original file", "Rows checked", "Exact overlaps", "Near overlaps"],
                [[name, result["rows"], result["exact_overlaps"], result["near_overlaps"]]
                 for name, result in verification["original_corpus"].items()]))
sections.append("### Annotated reviews not selected\n\n" + table(["Reason", "Count"], list(report["skipped"].items())))
doc = HERE / "CUSTOM_DATASET.md"
prefix = doc.read_text(encoding="utf-8").split("<!-- GENERATED_COUNTS -->")[0]
doc.write_text(prefix + "<!-- GENERATED_COUNTS -->\n\n" + "\n".join(sections), encoding="utf-8")

provenance = {(p["split"], p["row_index"]): p for p in read_jsonl("provenance.jsonl")}
output = {"train": read_jsonl("train_seed.jsonl"), "test": read_jsonl("eval_target.jsonl")}
with (HERE / "manual_spot_checks.tsv").open(encoding="utf-8", newline="") as handle:
    checks = list(csv.DictReader(handle, delimiter="\t"))
assert len(checks) == 20 and len({(c["split"], c["row_index"]) for c in checks}) == 20
rows = []
for check in checks:
    p = provenance[(check["split"], int(check["row_index"]))]
    row = output[check["split"]][p["output_row_1based"] - 1]
    rows.append([f"{p['split']}/{p['row_index']} (ID {p['source_id']})",
                 f"{p['file']}:{p['output_row_1based']}", row["input"], p["human_complaint"],
                 row["label"]["intent"], row["label"]["urgency"], row["label"]["product"],
                 row["label"]["sentiment"] + " / " + p["sentiment_rule"], check["assessment"], check["manual_note"]])
spot_doc = ("# Twenty-row manual spot-check\n\n"
            "The same Codex LLM annotator reread these full available source reviews against the written guidelines. "
            "This is LLM-assisted, same-annotator validation, not independent human review. "
            "Nineteen rows were confirmed; test/120 urgency was corrected to cao for explicit sớm. "
            "The table shows final rebuilt labels. Source row indexes are zero-based; output positions are one-based. "
            "Inputs equal the whitespace-normalized source review (with identifier masking where needed).\n\n")
(HERE / "MANUAL_SPOT_CHECK.md").write_text(spot_doc + table(
    ["Source split/index (ID)", "Output position", "Full review", "Human complaint", "Intent", "Urgency", "Verbatim product", "Sentiment / rule", "Assessment", "Manual review note"], rows), encoding="utf-8")
print(json.dumps({"counts": report["selected"], "skipped": report["skipped"],
                  "label_distributions": distributions, "domains": report["domains"],
                  "manual_spot_checks": len(checks), "overlaps": verification}, ensure_ascii=False, indent=2))
