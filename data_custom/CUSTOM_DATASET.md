# B2 custom dataset: UIT-ViOCD side experiment

This is the MOCK 5 side experiment. The existing `data/` corpus remains the
graded main run. No training, evaluation runs, adapters, or commits were made.
The custom files are `train_seed.jsonl` (250 rows) and `eval_target.jsonl` (50 rows).

## Source, use restrictions, and citation

Source: [tarudesu/ViOCD on Hugging Face](https://huggingface.co/datasets/tarudesu/ViOCD).
The dataset supplies Vietnamese reviews, human complaint labels (0/1), and four
domains: mobile, app, fashion, cosmetic. The source card restricts use to research;
it does not name a standard open-source license. Treat these derivatives as
research-only, including the local source snapshots. The repository's own license
does not override the dataset restriction. See the saved `source/README.md`.

Citation: Nguyen, Nhung Thi-Hong; Ha, Phuong Phan-Dieu; Nguyen, Luan Thanh;
Nguyen, Kiet Van; and Nguyen, Ngan Luu-Thuy (2021).
*Vietnamese Complaint Detection on E-Commerce Websites*.
[arXiv:2104.11969](https://arxiv.org/abs/2104.11969).
Use this title and eprint from the card's citation block; the card's introductory
paper link differs from that block.

The task describes 5,484 reviews. The downloaded train Parquet has **4,387** rows
and the test Parquet **549**. The card reports validation as 548, giving 5,484,
although its prose and the paper abstract say 5,485. This build uses the observed
train/test counts and never samples validation. The validation Parquet is not
downloaded or used.

Source repository revision recorded by the download:
`25e1851683ebc407d28e5a2d514b7ae0eadd4e4b`.
Parquet was downloaded with `.venv\Scripts\python.exe` through
[the source Parquet API](https://huggingface.co/api/datasets/tarudesu/ViOCD/parquet).
URLs, byte sizes, and SHA-256 checksums are in `source/download_manifest.json`.
`build_viocd.py` pins and checks both Parquet checksums before reading, so an
upstream change cannot silently reuse these annotations on different reviews.

## Sampling and split provenance

This is a deliberately curated convenience sample, **not a random sample or a
balanced benchmark**. A word-boundary product lexicon proposes candidates from
the source `review` field. It never assigns intent, urgency, sentiment, or the
final product span. The source's `review_tokenize` field is not used. There were
1,745 train candidates and 217 test candidates. The remaining 2,642 train and
332 test rows had no lexicon hint and were not part of the main shortlist; this
does not mean all of them lack valid products.

The main reading pool prioritizes human complaint label 1, plus label-0 candidates
containing `?`, `hỏi`, `mong`, or `cần`; candidates are displayed in source row
order. Explicit refund, exchange, and concrete information requests are also
reviewed from searches across the same training split to improve scarce-intent
coverage. These searches cannot assign a label. Each decision comes from reading
the full available review. Named spans beyond the hint lexicon, such as `gradpay`
and `meSenger`, may be chosen after reading.

`manual_annotations.tsv` is the authoritative ordered decision ledger. It contains
451 explicit decisions: 378 train and 73 test. The builder accepts eligible rows
in ledger order after decontamination, up to 250 train and 50 test. Later eligible
rows remain annotated reserves. Tests are processed first in this fixed ledger,
so a matching training candidate would be excluded rather than replacing an
already accepted test. There is no outcome-dependent sampling or model-score use.

Every training output comes only from source **train**; every evaluation output
comes only from source **test**. `provenance.jsonl` records the source split,
zero-based Parquet row index, original `Unnamed: 0` ID, domain, human label,
source-review hash, masking counts, output row, sentiment rule, and rationale.
The original ID alone is not used to identify rows across splits.

## Annotation guidelines and annotator

The annotator was the Codex LLM assistant in this session, reading Vietnamese
reviews and making explicit decisions. This is **LLM-assisted annotation with
stated rules and a manual second-pass spot-check by the same assistant**. The
original complaint labels were supplied by human source annotators; the new
intent, urgency, product, and sentiment labels were not independently annotated
or approved by a person. No inter-annotator agreement is claimed. The TSV stores
a per-review rationale; `annotations.jsonl` is its expanded machine-readable form.

Rules are operational mappings from reviews to the existing ticket label space.
A review need not contain an explicit support request if it reports an actionable
failure. Pure praise, vague insults, feature proposals, game-balance preferences,
moderation requests, or driver-attitude complaints without an allowed incident
are skipped. Uninterpretable text and missing target products are also skipped.

### Intent

| Label | Decision rule |
| --- | --- |
| `hoan_tien` | Explicit repayment/refund request, pending refund, or refusal to repay. Paid credits missing without requesting money back are a fault, not automatically a refund. |
| `doi_tra` | Explicit exchange/return request or a concrete ongoing exchange/return procedure dispute. Hypothetical general return-policy commentary alone is insufficient. |
| `van_chuyen` | Late, unfulfilled, cancelled, incomplete, mislabelled, or wrong-item/color/size shipment; missing advertised gifts/accessories; damaged or suspiciously opened delivery packaging. Ride availability and driver attitude alone are excluded. |
| `san_pham_loi` | Intrinsic quality, material, fit, breakage, adverse cosmetic reaction, or software/content malfunction. Includes crashes, failed account access, lost data, wrong answer keys, undismissable intrusive ads, failed purchased credits, and regression of previously available functionality. Intended gameplay balance and requested new features are excluded. |
| `hoi_thong_tin` | Concrete question about existing price, release timing, billing, warranty status, privacy, account operation, or learning guidance without a more specific remedy/fault taking precedence. A rhetorical question or proposal to add a feature is insufficient. |

For multiple issues, prioritize the latest explicit requested remedy: refund,
then exchange/return. Otherwise choose the concrete dominant incident: fulfilment
versus intrinsic fault. A question asking why an app is broken still receives
`san_pham_loi`; a factual question without established failure receives
`hoi_thong_tin`. Do not infer a refund or return just from dissatisfaction.

### Urgency

| Label | Decision rule |
| --- | --- |
| `cao` | Explicit expedited language such as `gấp`, `cần gấp`, `sớm`, `nhanh`, `nhanh chóng`, `mau chóng`, or `sớm nhất có thể`; a concrete unresolved failure/return/refund lasting at least a week; or described active harm such as repeated skin irritation/peeling or unauthorized account access. |
| `trung_binh` | Ordinary actionable unresolved delivery, payment, return, account, software, or product problem without the high-urgency cues. Strong anger or profanity alone does not raise urgency. |
| `thap` | Routine information question, mild nonblocking cosmetic/fit/texture/graphics/performance issue, or a politely discussed buyer-selected sizing exchange, without high-urgency evidence. |

Time estimates refer to the current unresolved incident, not how many years a
person has used an app. Severity and urgency are distinct: even a graphics-only
issue can be `cao` when an expedited fix is explicitly requested.

### Product

Choose a nonempty **case-sensitive verbatim span in the review** identifying
the target physical product, component/accessory/gift, or software/service. Prefer
a full explicit phrase or brand (`kem nền`, `ứng dụng line`, `hada labo`) when
available. Colloquial categories (`máy` in a phone purchase, `game`, `ứng dụng`,
`quần`) are allowed when the text clearly identifies the reviewed target. Named
marketplace services (`tiki`, `lazada`) may identify the service being criticized
for fulfilment/returns when the purchased physical item is unnamed. This is an
explicit service-product interpretation, not an inferred physical-item name.

Do not invent a product using the source domain. A phone mentioned only as the
device running an unnamed app is not that app's product; similarly, comparisons,
metaphors, insults, and previously purchased brands are not the current target.
Generic `hàng`, `sản phẩm`, `đơn`, and ungrounded plural `các ứng dụng` are
insufficient. Preserve case, misspellings, accents, and source anonymization.

### Sentiment: use the human complaint label first

| Rule | Written rule |
| --- | --- |
| S1 | Baseline: source complaint **1 → `tieu_cuc`**, **0 → `trung_tinh`**. Retain this unless a below override applies. Thanks, isolated praise, or politeness do not by themselves negate dissatisfaction with a failed remedy or material fault. |
| S2 | For complaint 1, override to `trung_tinh` only when the review explicitly accepts/recommends the product overall or gives a favorable conclusion/rating, with a mild nonblocking issue and no overriding material harm, failed remedy, or frustrated rejection. Sarcasm is not favorable approval. |
| S3 | Override to `tich_cuc` for explicitly favorable overall praise accompanying a routine factual question or polite buyer-selected exchange request, with no expressed dissatisfaction about a failed remedy. All five selected cases have human complaint 0. |
| S4 | For complaint 0, override to `tieu_cuc` if the review nevertheless explicitly reports dissatisfaction or a material failure. No selected row required S4. |

Selected rule uses: S1 **269**, S2 **26**, S3 **5**, S4 **0**. Overrides are
recorded in the ledger and provenance. Label-space values remain exactly those
in the original instruction.

## Text preservation, schema, and privacy

Read the first row of `data/train_seed.jsonl` as UTF-8. Copy its `instruction`
value verbatim and retain exactly its outer key order:
`instruction`, `input`, `output`, `label`. Retain the original label/output order:
`intent`, `urgency`, `product`, `sentiment`. `output` is exactly
`json.dumps(label, ensure_ascii=False)` with Python's default separators.
No provenance or extra annotation fields enter these two training-format files.

`input` comes directly from source `review`, changing only whitespace runs to
single spaces and masking matches for email addresses, Vietnamese phone numbers
(including separators and country codes), and contextual or recognizable order
codes, using `[EMAIL]`, `[PHONE]`, and `[ORDER_CODE]`. Order-context masking precedes
phone matching. Do not correct spelling, remove profanity/emojis, rewrite
requests, fill truncated text, or normalize the product's case/accents. The
selected product must survive masking and appear in the normalized original.

The 300 selected reviews contain **0 detected emails, 0 detected phone numbers,
and 0 detected order codes** under these patterns; no replacement was needed.
Dates, prices, sizes, product/model numbers, account IDs, and ordinary numbers are
not automatically treated as phone/order identifiers. This regex policy does not
claim to anonymize every possible personal name or identifier. Raw source and
unselected candidate snapshots should retain research-only handling.

## Decontamination and verification

1. Normalize whitespace and mask identifiers. For comparison only, apply Unicode
   NFKC, case folding, and extract Unicode word tokens; join them with spaces.
   This ignores punctuation/emoji differences and preserves Vietnamese accents.
2. Exact equality of this canonical text is a duplicate. A lexical near-duplicate
   is a character `SequenceMatcher` ratio **≥ 0.88** (`autojunk=False`), or a
   **3-word-shingle Jaccard similarity ≥ 0.80** when both texts have at least six
   tokens. Character comparison also covers short reviews. Length and quick-ratio
   pruning cannot discard pairs that could pass the stated character threshold.
3. For every candidate, compare against every accepted custom row from either
   split and every input in every `data/*.jsonl` file. Skip matches. The normalized
   source duplicate train row **858** was excluded against train row **674**.
   This was exact; no additional near-only rejection occurred.
4. Verify the final files exhaustively again: **44,850** custom row pairs
   (including **12,500** train/eval pairs), plus **100,500** comparisons against
   all **335** original corpus inputs. Check source split, index/ID/hash, raw-to-
   input transformation, schema, instruction, label space, output serialization,
   product spans, and detectable identifiers.
5. Hash every file under `src/`, `notebooks/`, `scripts/`, `tests/`, `data/`,
   `results/`, and `adapters/` before source download and after the build. All
   protected files are unchanged. This side experiment does not alter the main run.

Final exact overlaps: **0**. Final lexical near overlaps: **0**, both within the
custom splits and between train/eval. Final overlap with each original file is
**0 exact + 0 near**, explicitly including all 15 rows in
`data/eval_regression.jsonl`. Full per-file results are below and in
`verification_report.json`. These checks do not claim semantic paraphrase
detection or decontamination against unknown external corpora.

Of 451 explicitly decided reviews, **151 were not selected**: 90 unsupported
intents, 19 lacking a target product, 18 pure praise without a request, 6 ambiguous
texts, 1 duplicate, and 17 eligible reserves after the quota filled. Thus **108**
were skipped for having no supported intent, including pure praise. These counts
are separate from the earlier lexicon-screen exclusions, which were not fully
annotated. The ledger and `rejections.json` preserve the distinctions.

## Manual spot-check

Twenty final rows were reread by the same LLM annotator, covering all five intents,
all three urgency/sentiment labels across the sample, all four source domains,
sentiment overrides, explicit remedies, service products, and ambiguous noisy
size language. **19 labels were confirmed; 1 urgency label was corrected**:
test source row 120 changed from `thap` to `cao` because it explicitly asks for an
early fix (`sớm`). The corrected dataset was rebuilt and verified. The evidence,
current labels, source IDs, output positions, and per-row review notes are in the
20-row [manual spot-check table](MANUAL_SPOT_CHECK.md), with source notes in
`manual_spot_checks.tsv`. This is not independent human validation.

## Why this is distributionally new

The graded main corpus uses synthetic ticket templates, explicit product phrases,
and invented order-code boilerplate. These reviews are real source user text with
varied word order, informal spelling, missing diacritics, stylized app names,
emojis, profanity, long mixed praise/complaint passages, source truncation, and
no generated order-code template. Inputs were neither paraphrased nor made to
look like support tickets. Software/content faults, account problems, cosmetics,
garment fit, and promotional-gift fulfilment differ from the synthetic phrasing.
The preserved instruction and label space allow a controlled input-distribution
side experiment; the main corpus remains the graded comparison.

## Limitations

- Curated, complaint-enriched convenience sampling creates strong class/domain
  imbalance: 294/300 source complaint labels are 1. Software reviews dominate.
  The test set has **no `tich_cuc` rows**, so positive-sentiment performance cannot
  be estimated from this evaluation file. Scarce refund/information counts also
  limit per-class conclusions. No balanced or population-representative result
  is claimed.
- ViOCD was annotated for complaint detection, not ticket intents or urgency.
  Mapping implicit review complaints to support intents is subjective; the labels
  are LLM decisions with a same-annotator check, not new human ground truth.
- Generic product spans and named marketplace services have less physical-item
  specificity than the main corpus. Product filters miss some eligible reviews.
- The source `review` field already contains preprocessing artifacts, brand
  placeholders, strange substitutions, and truncated text. It is preserved as
  supplied, not claimed to be untouched historical raw posts.
- Lexical duplicate thresholds miss some semantic paraphrases. Identifier regexes
  cover the required common phone/email/order patterns, not all personal data.
- This task builds a dataset only. It supplies no fine-tuning or model-performance
  claim; research-only restrictions apply to future use.

## Reproduction and artifacts

From the repository root on Windows:

```powershell
.venv\Scripts\python.exe -m pip install --no-cache-dir --target data_custom/_vendor pyarrow==24.0.0
.venv\Scripts\python.exe data_custom/build_viocd.py download
.venv\Scripts\python.exe data_custom/build_viocd.py candidates
.venv\Scripts\python.exe data_custom/build_viocd.py build
.venv\Scripts\python.exe data_custom/write_report.py
```

For the saved, checksum-verified source snapshots, `build` and `write_report.py`
work offline. An optional `verify` mode rechecks an existing build. Only
`data_custom/` is written; the Parquet dependency is isolated in the locally
ignored `_vendor/` directory. The protected-file baseline is preserved on rebuild.
Source changes require reannotation rather than silently overriding checksums.

Files include the two JSONL outputs; builder; explicit TSV and expanded JSONL
annotations; source snapshots and manifests; candidate snapshots and counts;
per-output provenance; skip reasons; build and verification reports; the protected
file-hash baseline; this document; and the 20-row manual spot-check table.

<!-- GENERATED_COUNTS -->

## Final counts and distributions

| Output | Source split | Rows |
| --- | --- | --- |
| train_seed.jsonl | train | 250 |
| eval_target.jsonl | test | 50 |

### intent

| Label | Train | Eval |
| --- | --- | --- |
| doi_tra | 13 | 2 |
| van_chuyen | 50 | 15 |
| hoan_tien | 6 | 1 |
| san_pham_loi | 172 | 31 |
| hoi_thong_tin | 9 | 1 |

### urgency

| Label | Train | Eval |
| --- | --- | --- |
| cao | 18 | 4 |
| trung_binh | 197 | 42 |
| thap | 35 | 4 |

### sentiment

| Label | Train | Eval |
| --- | --- | --- |
| tieu_cuc | 221 | 47 |
| trung_tinh | 24 | 3 |
| tich_cuc | 5 | 0 |

### Source domains

| Domain | Train | Eval |
| --- | --- | --- |
| mobile | 15 | 2 |
| app | 149 | 26 |
| fashion | 50 | 10 |
| cosmetic | 36 | 12 |

### Product spans (complete distribution)

| Verbatim span | Train | Eval |
| --- | --- | --- |
| aP | 15 | 1 |
| aP hack não | 1 | 0 |
| bluezone | 1 | 0 |
| băng đô | 1 | 0 |
| chai xịt | 1 | 0 |
| cửa hàng E | 0 | 1 |
| dép | 13 | 1 |
| dầu gội | 1 | 0 |
| facebOk | 4 | 0 |
| facebook | 5 | 0 |
| facebook lite | 2 | 0 |
| game | 60 | 9 |
| game liên quân | 0 | 1 |
| grab | 1 | 0 |
| gradpay | 1 | 0 |
| hada labo | 2 | 0 |
| ins | 1 | 0 |
| instagram | 1 | 0 |
| kem | 4 | 6 |
| kem chống nắng | 1 | 0 |
| kem dưỡng trắng | 1 | 0 |
| kem nền | 5 | 1 |
| kem trị mụn | 1 | 0 |
| lazada | 10 | 0 |
| line | 1 | 1 |
| lords mobile - gamota | 1 | 0 |
| meSenger | 1 | 0 |
| máy | 7 | 0 |
| nước tẩy trang | 2 | 0 |
| phần mềm | 3 | 1 |
| quần | 16 | 6 |
| son | 9 | 2 |
| sách | 1 | 0 |
| sữa rửa mặt | 3 | 2 |
| tai nghe | 3 | 0 |
| tiki | 5 | 4 |
| tiki now | 1 | 0 |
| tinh chất | 2 | 1 |
| váy | 3 | 0 |
| zalo | 7 | 0 |
| zing mp3 | 1 | 0 |
| áo | 9 | 2 |
| áo quần | 1 | 0 |
| điện thoại | 4 | 1 |
| điện thoại samsung galaxy m21 | 0 | 1 |
| đầm | 8 | 1 |
| ứng dụng | 29 | 7 |
| ứng dụng line | 0 | 1 |
| ứng dụng zingmp3 | 1 | 0 |

### Original corpus decontamination

| Original file | Rows checked | Exact overlaps | Near overlaps |
| --- | --- | --- | --- |
| eval_regression.jsonl | 15 | 0 | 0 |
| eval_target.jsonl | 50 | 0 | 0 |
| holdout_secret.jsonl | 20 | 0 | 0 |
| train_seed.jsonl | 250 | 0 | 0 |

### Annotated reviews not selected

| Reason | Count |
| --- | --- |
| no_supported_intent | 90 |
| no_target_product | 19 |
| pure_praise_no_request | 18 |
| ambiguous_text | 6 |
| custom_duplicate | 1 |
| quota_full | 17 |
