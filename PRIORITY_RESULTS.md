# Priority 1–3 Results: Coverage, Competitor Sources, and Generalization

Run after improving extraction depth, adding peer sources, and validating that a
second company works by configuration alone.

---

## 1. Claim distribution by category

| Category | AMAT | Lam | Target | AMAT (before) |
|---|---:|---:|---:|---:|
| Products | 3 | 11 | 3 | **2** |
| Process steps | 3 | 4 | 2 | 0 |
| Technologies | 9 | 2 | 2 | 5 |
| Customer needs | 16 | 2 | 2 | 11 |
| Market drivers | 20 | 13 | 2 | 15 |
| Competitors | 6 | 4 | 1 | 0 |
| Commercial implications | 28 | 26 | 2 | 11 |
| **Total claims** | **105** | **83** | | 66 |
| Peer-filing claims (separate ledger) | 27 | 13 | | 0 |

Ledger depth is one thing; how much of it reaches the report is another. A first
synthesis pass collapsed 16 AMAT customer claims into a single generic entry. Adding
explicit section-density guidance ("one entry per distinct customer category the
ledger supports") moved it to four separate categories - foundry and logic, DRAM,
NAND, and flash - each with its own requirements and pain points, which is what the
filing actually discloses and what a product-marketing reader needs. Section output
after the change:

| Section | AMAT | Lam |
|---|---:|---:|
| Customers | 3 | 3 |
| Market drivers | 3 | 3 |
| Commercial insights | 2 | 4 |
| Competitors | 2 | 2 |

Every category is now at or above target for both companies, and the distribution
spread narrowed considerably: AMAT previously had 15 market-driver claims against
2 product claims.

**The targets were advisory and were never enforced.** Extraction is still gated on
a verifiable verbatim quote, so a category can legitimately come in short. The
mechanism that improved balance was measuring coverage after a first broad pass,
then running a second pass on the highest-value chunks with a hint naming only the
categories still below target. Because the quote gate is unchanged, the hint can
redirect attention but cannot cause an unsupported claim to survive.

---

## 2. Source distribution

Both companies: **22 sources, 22 retrieved, 0 failures.**

| Kind | AMAT | Lam |
|---|---:|---:|
| 10-K | 5 | 5 |
| 10-Q | 4 | 4 |
| 8-K (cover pages) | 8 | 8 |
| **Earnings releases (8-K Ex. 99.1)** | **4** | **4** |
| SEC XBRL company facts | 1 | 1 |
| Peer filings (Lam + KLA) | 2 | 2 |

The earnings releases are the material addition. An 8-K's primary document is only
a cover page; the substance is Exhibit 99.1, which was previously never fetched.
Applied Materials' Q3 FY2026 release alone contains "AI" 59 times, "advanced
packaging" 4 times and "HBM" 3 times, plus direct CEO commentary — commercial
language the 10-K simply does not contain.

**A finding that reframed the problem:** the product gap was never an extraction
failure. AMAT's Form 10-K contains the word "deposition" **once**, "etch" **once**,
and names **no product** and **no competitor** anywhere in 2.2 MB of HTML. Adding
more chunks from the 10-K could not have helped; the source lacked the content.

---

## 3. Unsupported / failed extraction cases

| | AMAT | Lam |
|---|---:|---:|
| Coverage shortfalls | 0 | 0 |
| Claims discarded by the quote gate | 11 | 11 |
| Companies with no segment table parsed | 0 | **1** |
| Peers with no recognisable earnings exhibit | 0 | 4 |

Two honest gaps remain:

**Lam's segment revenue table is not parsed.** Lam reports one reportable segment
and disaggregates revenue between systems and customer support. An implementation
attempt was made and **reverted**: Lam presents that table in *thousands* while its
geographic table is in *millions*, its column headers split across four lines, and
it has no labelled total row, so the row collector ran on into the following
gross-margin table and produced a segment figure of $11,725,308M — wrong by six
orders of magnitude and presented with the same confidence as a correct figure.
Reporting the table as absent is the better failure. The rationale is recorded in
`sec_tables.py`.

**Four of Lam's earnings 8-Ks still yield no exhibit.** The exhibit-name matcher
was generalised to handle Lam's `lrcx_exhibitx991xq4x2026.htm` style (`x` as
separator), which fixed the two most recent filings; four older ones still do not
match and are reported as warnings.

---

## 4. Was any company-specific code required?

**No.** Adding Lam Research required **zero** changes to Python logic.

Every match for "Lam" in the codebase is a comment or a prompt example:

```
semicon/chunk.py:387          a comment explaining a scoring bug
semicon/sources/sec_tables.py:75,97   comments explaining a documented gap
semicon/extract.py:367        an example inside the peer-extraction prompt
```

Lam exists only in YAML:

```yaml
# config/companies.yaml
- name: "Lam Research"
  ticker: "LRCX"
  sector_role: "equipment"
  relevant_process_steps: ["Etch", "Deposition", "CMP", "Advanced Packaging"]
```

Two *general* improvements were made while adding Lam, both of which help any
company rather than Lam specifically:

1. **Inline XBRL header stripping.** KLA's filing carried 253,842 characters of
   machine-only `ix:header` taxonomy that never renders but was competing with real
   prose for extraction budget.
2. **PART-boundary section fallback.** KLA's 10-K lists every item in its table of
   contents but runs the body under bare `PART I` / `PART II` markers, so an
   Item-only scan found no sections at all and the filing yielded nothing.

Both are format variations, not company special-cases: any filer using Inline XBRL
or PART-only headings benefits.

---

## 5. API cost per company

Measured on fresh runs with the LLM cache cleared.

| | AMAT | Lam |
|---|---:|---:|
| `gpt-4o-mini` calls (extraction + peers) | 45 | 28 |
| `gpt-4o` calls (synthesis) | 1 | 1 |
| Input tokens | 61,249 | 39,449 |
| Output tokens | 15,090 | 11,134 |
| **Total tokens** | **76,339** | **50,583** |
| Extraction cost | $0.0158 | $0.0106 |
| Synthesis cost | $0.0412 | $0.0330 |
| **Total cost** | **$0.0570** | **$0.0437** |
| Wall-clock | 2m33s | 2m08s |
| Repeat run | **$0.00** | **$0.00** |

Roughly **$0.05 per company**, about **5 cents**. Synthesis is 6–8x the cost per
token of extraction but runs once; extraction is spread across 28–45 calls.

---

## 6. Remaining weaknesses

1. **Segment/geographic parsing generalises less well than the rest.** It works
   for AMAT across three fiscal years, and fails for Lam. The parser assumes
   millions and a labelled total row. Proper unit detection (`in thousands` vs `in
   millions`) and stricter table-boundary handling would fix Lam and likely most
   other filers.
2. **Competitor analysis is evidence-gated, so it is only as good as the peers
   configured.** For AMAT it now names Lam Research and KLA with real product
   overlap, sourced from those companies' own filings. It does not discover peers,
   and it will not surface a competitor that is not in `settings.yaml`. A peer not
   on that list is invisible.
3. **Third-party industry sources are not automated.** SEMI, Yole and TechInsights
   are paywalled or restrict automated retrieval, so they remain a manual, cited
   addition. Only first-party EDGAR sources are automated.
4. **Earnings call transcripts are still out of scope** for the same terms-of-service
   reason; earnings *releases* now cover much of the same ground.
5. **Coverage targets are global, not sector-aware.** A foundry and an EDA vendor
   should not be measured against the same product-count expectation. Targets are
   currently one set for every company.
6. **`customer_needs` is thin for Lam (2) and rich for AMAT (18).** This tracks
   disclosure rather than analysis quality, but it means cross-company comparison of
   that section is not yet meaningful.
7. **The coverage pass roughly doubles extraction cost.** It stops early once
   targets are met, but on AMAT it still ran 17 chunks. A cheaper strategy would
   route only the most promising chunks, or reuse the first pass's per-chunk scores.
