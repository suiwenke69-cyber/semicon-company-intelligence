# Semiconductor Company Intelligence Agent

Turns a semiconductor company name into a structured, source-grounded intelligence
report. Built for semiconductor B2B roles — Product Marketing, Market Intelligence,
Business Development, Technical Sales, Strategic Marketing, Supply Chain — where
the question is **Technology → Product → Market → Customer → Competition → Business**.

Not an investment tool. It produces no price targets and no recommendations.

```bash
python main.py "Applied Materials"
```

---

## Status

| Stage | Engine | State |
|---|---|---|
| 1. Resolve company → CIK | Python | ✅ done |
| 2. Retrieve filings + XBRL | Python | ✅ done |
| 2c. Segment / geographic revenue tables | Python | ✅ done |
| 3. Chunk + select + extract claims | LLM (`gpt-4o-mini`) | ✅ done |
| 4. Analyze (drivers, competitors, insights) | LLM (`gpt-4o`) | ✅ done |
| 5. Validate citations | Python | ✅ done |
| 6. Render `company_report.md` | Python | ✅ done |
| 7. Generalise to other companies | config | ✅ done |

Verified across **two** equipment companies (Applied Materials, Lam Research), with
Lam requiring zero Python changes. See `PRIORITY_RESULTS.md` for the full
comparison: claim distribution, source distribution, cost per company, and
remaining weaknesses.

All stages are implemented. Two commands:

```bash
python main.py "Applied Materials"            # deterministic only, zero tokens
python main.py "Applied Materials" --analyze  # adds the LLM stages
```

**The LLM stages never run by accident.** Retrieval and financials are a separate
code path, and `--analyze` is required before the pipeline will construct a model
client at all.

### Verified

The LLM stages were first developed against an injected stub client, then run
against the live OpenAI API on Applied Materials. Measured, not estimated:

| | |
|---|---|
| Extraction calls (`gpt-4o-mini`) | 45 |
| Synthesis calls (`gpt-4o`) | 1 |
| Tokens | 76,339 |
| **Cost for a full company report** | **$0.057** (Lam: $0.044) |
| Validated claims in the ledger | **109** |
| Analytical categories at target | **7 / 7** |
| Dangling citations | **0** |
| Repeat run | **$0.00** |

Extraction is the cheap part ($0.0088 for 16 calls). Synthesis is 6x the cost per
token but runs once. A 40,000-character ceiling was set in advance; the run used
39% of the token budget.

**Note on extraction-prompt changes:** prompts are cached by content hash, so
editing `SYSTEM_PROMPT` is the one action that re-bills. A prompt change is a new
experiment; a re-run is free.

### The anti-hallucination gate, in practice

On the live run the gate discarded fabricated claims while keeping genuine ones,
and validation reported zero dangling citations across 66 claims — meaning every
analytical item in the report traces to a claim that traces to a filing.

---

## Architecture

```
Input
  → resolve     name → CIK                        [Python]
  → retrieve    filings + XBRL into cache/        [Python]
  → extract     text → Evidence[] (cited claims)   [LLM, gpt-4o-mini]
  → analyze     Evidence[] → analysis              [LLM, gpt-4o]
  → validate    citations must resolve             [Python]
  → render      profile → company_report.md        [Python]
```

Six stages, one process, no agents.

Three design decisions carry the whole thing:

**1. No LLM ever touches a number.** Revenue, R&D, margins and EPS come from SEC's
XBRL `companyfacts` API as exact integers with an accession number attached.
Segment and geographic revenue are parsed from the 10-K MD&A with plain Python
table parsing. This removes the largest hallucination surface and roughly 60% of
the context a "paste the 10-K" approach would need.

**2. The model may only report what it can quote.** Every extracted claim must
carry a verbatim quote, and that quote is checked against the source text in
Python. A claim whose quote cannot be found is **discarded, not softened**. This
turns the specification's warning about hallucinated semiconductor information
into an enforced gate rather than a prompt request.

**3. Analysis reads claims, not documents.** `evidence.json` is the contract
between extraction and analysis. Analysis never sees a filing; it receives a
compact ledger and may only cite `claim_id`s that exist. Validation is therefore a
set-membership test, and a fabricated product or customer is caught by an
assertion rather than by a human re-reading the 10-K.


---

## What is verified working

Against Applied Materials (CIK 0000006951), 15/15 filings retrieved, 17 artifacts
cached, **0 network requests on a repeat run**.

Financial totals were cross-checked against the 10-K text rather than trusted from
the parser:

| Metric | XBRL value | 10-K text |
|---|---|---|
| FY2025 net revenue | $28,368M | `Net revenue $ 28,368 $ 27,176 $ 26,517` |
| FY2025 R&D | $3,570M | `Research, development and engineering (RD&E) $ 3,570 $ 3,233 $ 337` |
| FY2025 diluted EPS | $8.66 | `Earnings per diluted share $ 8.66 $ 8.61 $ 0.05` |

Independently, the eight constituent geographic regions sum to **$28,368M**,
reconciling exactly with the XBRL revenue figure by a completely separate code
path. That cross-check is the strongest evidence the deterministic layer is sound.

Example output:

```
Semiconductor Systems      20,798M   73%    +4.5% YoY   FY2025
Applied Global Services     6,385M   23%    +2.6% YoY   FY2025
Corporate and Other         1,185M    4%   +13.9% YoY   FY2025

China                       8,529M   30%   -15.7% YoY
Taiwan                      6,857M   24%   +71.0% YoY
Asia Pacific               24,343M   86%   +11.1% YoY   [subtotal]
```

---

## Bugs found and fixed during development

Recorded because they are the interesting part, and because each now has a
regression test.

1. **Year-over-year compared a quarter against a year.** Operating cash flow
   reported `-8.3% YoY` by comparing a 90-day figure ($1,686M) with a 363-day one
   ($7,958M). Worse than reporting nothing. Fixed by enforcing matching period
   length; see `_compute_yoy`.

2. **XBRL's `fy` element describes the filing, not the period.** The prior-year
   comparative inside a FY2024 10-K carries `fy=2024` while ending in October
   2023, producing two different periods both labelled `FY2024`. Period labels
   are now derived from the period end date plus the issuer's fiscal-year-end
   month.

3. **A flat period cap evicted recent quarters.** Capping 12 periods across
   annual *and* quarterly facts pushed out every recent quarter for metrics with
   long annual history. Annual and quarterly history are now capped separately.

4. **A column header parsed as a data row.** `2025 2024 2025 over 2024` was read
   as a row labelled "over" with four amounts, corrupting the parts-sum-to-total
   check and causing the *correct* rows to be discarded.

5. **Over-strict row filters discarded real data.** A magnitude threshold
   (`>= 1000`) silently dropped AMAT's genuine `Display` ($885M) and
   `Corporate and Other` ($155M) segments; a word-length rule dropped the region
   `China`. Both heuristics were replaced with structural checks.

6. **Subtotals were double-counted.** `Asia Pacific` is the sum of five regions
   above it, so summing all rows made a correct parse look broken. Subtotals are
   now flagged and excluded from validation.

7. **Name normalisation destroyed company identity.** Stripping the token
   `" technologies"` collapsed "Micron Technology" to "micron", so the fuzzy
   matcher rejected the near-miss "Micron Tecnology" at 0.545 similarity. Now
   only legal designators and state-of-incorporation suffixes are stripped.

8. **A metric's note described its pre-truncation state**, advertising "18
   annual, 18 quarterly" for a metric holding 12 and 12.
9. **A "Signatures" terminator truncated the whole document.** Intended to stop
   parsing before the exhibit index, it instead matched the *table-of-contents*
   entry near the top, discarding the entire filing body. The length filter
   already handled contents entries, so the terminator was purely harmful.
10. **An item-heading regex required a non-space after the separator.**
    AMAT renders `Item 1:&#160;&#160;&#160;Business`, so headings arrived padded
    with spaces and matched nothing. Zero sections were found.
11. **Money formatting was implemented twice** and the per-share case was fixed in
    the CLI but not the renderer, so the report printed an EPS of 8.66 as "9".
    Both now call `formatting.format_money`.
12. **Near-duplicate removal did not exist.** Only exact content hashes were
    compared, so lightly edited repeats were sent to the model twice, wasting
    tokens and inflating how many independent sources appeared to support a claim.
13. **A length guard in the dedupe compared set sizes to word counts**, so a
    63-word near-duplicate was rejected as "too short to compare" against a
    60-word chunk. It now compares like with like.
14. **Validation warned about balance-sheet metrics having no duration.** Correct
    behaviour for `inventory` and `deferred_revenue` was reported as a defect,
    producing a permanent warning that would train a reader to ignore warnings.
15. **The budget guard aborted affordable runs halfway.** It reserved
    `max_output_tokens` (4,000) as *expected* output per call, so it refused to
    start chunk 12 of 18 on a run that actually used 39% of budget. It now
    projects with a realistic completion size while `max_output_tokens` continues
    to cap each individual call.
16. **The model's segments overwrote the deterministically parsed ones.** The
    pipeline computed $20,798M for Semiconductor Systems, then replaced the
    segment with the model's narrative object — which has no revenue field — so
    the report rendered the whole segment table as "n/a" while the correct numbers
    sat unused one layer below. Enrichment and parsing are now merged explicitly.
17. **A filing source id was validated as if it were a claim id.** Deterministic
    table parses cite the *filing* (`sec_10_k_2025`); LLM items cite *claims*
    (`c_...`). Sharing one field made the validator report a correct parse as a
    dangling citation. The two kinds of provenance are now separate fields with
    separate checks.
18. **The extraction prompt suppressed all risks and competitors.** It instructed
    the model to ignore "risk-factor genericities", and the model over-applied
    that to every risk: on an AMAT 10-K whose Risk Factors section plainly names
    customer concentration, geographic concentration and export controls,
    extraction returned **zero** risk claims and **zero** competitor claims. After
    rewording, the same filing yields 6 risk and 7 competitor claims, and the
    ledger grew from 27 to 66 claims.

Defects 16-18 were found only by running against the live API — the offline suite
passed throughout. They are locked down in `tests/test_live_run_regressions.py`.

Run `pytest` to confirm all of these stay fixed. The suite is 114 tests, fully
offline and free to run.

---

## Token efficiency

| Lever | Mechanism |
|---|---|
| Numbers never enter a prompt | XBRL + deterministic table parsing |
| Repeat runs cost $0 | Content-addressed HTTP cache **and** LLM response cache |
| Only relevant text sent | Item-level section selection, not whole filings |
| Budgeted before spending | Per-section and total character ceilings in `chunk.py` |
| No duplicated context | Exact-hash **and** near-duplicate (Jaccard) chunk removal |
| Cheap model where possible | `gpt-4o-mini` extraction, `gpt-4o` synthesis only |
| Ledger stays small | Claims are 1-sentence assertions, not prose |
| Profiles are the unit of reuse | `data/profiles/*.json` committed, diffable |
| No accidental overspend | `run_token_budget` aborts at the ceiling |

Measured on Applied Materials: 324,748 characters of 10-K text become **29,914
characters (~7,500 tokens)** of extraction context across 16 chunks, at
`gpt-4o-mini` rates. A cold run makes ~17 HTTP requests; a warm run makes none.

### Running the LLM stages

Create a `.env` file in the project root (it is gitignored):

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-your-key-here
EOF
```

Or export it in your shell instead. A real environment variable takes precedence
over `.env`, because a shell export is an explicit act by the operator and should
not be silently overridden by a file in the checkout.

```bash
python main.py "Applied Materials" --analyze
```

`--preview` shows exactly how much text would be sent, with no API call at all.
The run prints per-model token counts and dollar cost, and aborts rather than
exceeding `llm.run_token_budget`.

### Browsing the reports

```bash
python main.py --web           # regenerate site/ from all saved profiles
python main.py --web --open    # ...and open it in the browser
```

Produces a **static HTML site** in `site/`: an index dashboard comparing every
company, plus one self-contained page per company with all seven framework
sections and the searchable Evidence Ledger.

Deliberately static — no server, no new dependency, no build step. Each page
inlines its CSS, so a single HTML file can be sent to someone as-is. The tradeoff,
stated plainly: a static site cannot trigger a new analysis, which is the right
trade for a portfolio artefact because the interesting thing to show is the output
and its traceability, not a button that spends money.


---

## Adding a company

Edit `config/companies.yaml` only. No code changes.

```yaml
- name: "Lam Research"
  ticker: "LRCX"
  cik: null                    # null -> looked up from SEC's ticker map
  sector_role: "equipment"     # equipment|materials|foundry|idm|fabless|eda|osat
  relevant_process_steps: ["Etch", "Deposition", "CMP"]
```

Foreign private issuers work without special-casing: ASML files `20-F`, which is
already in `filings.annual_forms` alongside `10-K` and `40-F`.

---

## Layout

```
main.py                        CLI
config/settings.yaml           runtime knobs, budgets, HTTP policy, model choice
config/companies.yaml          the only file to edit to add a company
semicon/models.py              schemas: Evidence, CompanyProfile, financials
semicon/chunk.py               section discovery, chunking, dedupe, budgeting
semicon/extract.py             stage 3: chunks → verified claims (quote gate)
semicon/analyze.py             stage 4: claims → analysis (citation-constrained)
semicon/validate.py            stage 5: mechanical citation and rule checks
semicon/render.py              stage 6: Jinja → markdown
semicon/formatting.py          shared display formatting (CLI + report)
semicon/llm/client.py          cached, budget-capped, provider-pluggable client
semicon/resolve.py             company name → CIK
semicon/http.py                paced, retrying, content-addressed HTTP cache
semicon/pipeline.py            stage orchestration
semicon/sources/sec_edgar.py   filing discovery + HTML→text
semicon/sources/sec_financials.py  XBRL → exact metrics
semicon/sources/sec_tables.py  MD&A segment/geographic table parsing
templates/report.md.j2         report template
cache/                         downloaded artifacts + LLM responses (gitignored)
data/profiles/                 extracted profiles (committed)
tests/                         offline regression suite (98 tests)
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python main.py "Applied Materials"
.venv/bin/pytest
```

Set `user_agent_contact` in `config/settings.yaml` before running at any volume:
SEC's fair-access policy requires a descriptive User-Agent with real contact
information, and requests without one are rejected with HTTP 403.

## Known limitations

- **Segment/geographic data needs an annual report.** The XBRL company-facts API
  omits dimensionally tagged facts, so these come from 10-K MD&A tables. A
  company that phrases its segment table unusually yields a warning, not a guess.
- **AMAT's FY2023 segment table is not parsed** — that filing's MD&A does not
  present a "Net revenue by segment" table in the expected form, so the
  deterministic parser reports a warning. The geography table for the same year
  parses fine. The FY2025 figures, which the report leads with, are unaffected.
- **Company product pages are not scraped.** They return HTTP 403 to plain HTTP
  clients. The design does not depend on them.
- **Earnings call transcripts are out of scope.** Free transcripts sit behind
  terms-of-service restrictions; the 8-K press release and 10-K MD&A are used
  instead as legitimate substitutes.
- **Process-step mapping is bounded by a closed taxonomy.** The model selects from
  `ProcessStep` (and a company's configured subset); an invented step is dropped
  rather than coerced onto the nearest real one.
- **Competitor lists are short when the filing is vague, and that is by design.**
  AMAT's 10-K names no competitors at all — it describes rivals generically. The
  report says so explicitly rather than padding the list from industry knowledge.
  Competitor identification from industry sources is a documented future
  extension, deliberately not guessed at.
- **Segment and geographic figures come from the newest annual report that
  parsed**, so a segment last reported in FY2024 (AMAT's Display) is labelled with
  its year rather than presented as current.


## Source priority

1. Annual reports (10-K / 20-F / 40-F)
2. Quarterly reports (10-Q)
3. 8-K earnings press releases
4. DEF 14A proxy statements
5. SEC XBRL company facts
6. Investor presentations (flag-gated, best-effort)

Low-quality SEO aggregators are not used.
