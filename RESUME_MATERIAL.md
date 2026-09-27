# Resume Material: Semiconductor Company Intelligence Agent

Everything here is drawn from the actual repository and is factually checked
against it. Figures are the measured ones, not rounded up for effect — a resume
claim that does not survive a follow-up question in an interview is worse than no
claim at all.

**Repo:** https://github.com/suiwenke69-cyber/semicon-company-intelligence

---

## Use this prompt in ChatGPT

Copy the block below, then paste the **"Context block"** underneath it. That gives
ChatGPT everything it needs without it having to guess or invent numbers.

```
You are helping a graduate student in Semiconductor Technology and Operations
tailor a resume for semiconductor B2B commercial roles: Product Marketing, Market
Intelligence, Business Development, Technical Sales, and Strategic Marketing.

Below is a verified project context block. Rewrite it into:
1. One project entry formatted for a resume (project title, 3-4 achievement
   bullets, tech stack line). Start each bullet with a strong action verb and
   include a concrete number wherever the context provides one.
2. A 2-sentence version for a LinkedIn "Projects" section.
3. One STAR-format interview answer (Situation, Task, Action, Result), about 150
   words, suitable for "tell me about a technical project you led".

Constraints:
- Use ONLY the numbers and facts in the context block. Do not invent metrics,
  users, revenue impact, or team size.
- This is a solo project. Do not imply a team.
- Do not claim investment analysis, financial advice, or valuation work — the
  project deliberately avoids those.
- Emphasise what is relevant to the target roles: understanding how a
  semiconductor company makes money, mapping technology to products and markets,
  customer segmentation, competitive positioning, and evidence-based reasoning.
- Distinguish clearly between the evidence-gated claims the pipeline verifies and
  the analytical inferences it produces.
- Keep the tone technical but readable for a non-engineer recruiter.
```

---

## Context block (paste this into ChatGPT after the prompt)

```
PROJECT: Semiconductor Company Intelligence Agent (solo, Python)

WHAT IT DOES
Takes a semiconductor company name and produces a structured intelligence report
following the chain: Technology -> Product -> Market -> Customer -> Competition ->
Business. Supports Applied Materials, Lam Research, KLA, ASML, Micron, NVIDIA, AMD,
Infineon, NXP and Analog Devices. Adding a new company requires editing a YAML
config file only — no Python changes.

SCALE (measured)
- 10 semiconductor companies analysed
- 656 evidence-gated claims extracted across those companies
- 175 cited first-party source documents
- 6,838 lines of Python across 21 modules
- 122 offline tests, all passing, costing nothing to run

CORE TECHNICAL DESIGN
1. No language model ever touches a number. Revenue, R&D, margins and EPS are
   pulled from SEC XBRL as exact values with accession-level provenance. Segment
   and geographic revenue are parsed from 10-K MD&A tables with deterministic
   Python table parsing, not by a model.
2. Anti-hallucination gate: every qualitative claim extracted by the model must
   carry a quote copied verbatim from the source filing. The quote is checked
   against the source text in Python; a claim whose quote cannot be found verbatim
   is discarded, not softened. On a live test, all 48 deliberately fabricated
   claims were rejected while genuine claims were retained.
3. Citation-constrained synthesis: the analysis stage never sees a filing. It
   receives only a claim ledger and may cite only claim IDs that exist, which turns
   citation validation into a programmatic check. Result across all runs: zero
   dangling citations.

ARCHITECTURE
Input -> resolve (name to CIK) -> retrieve (SEC filings + XBRL) -> extract (chunk to
claims) -> analyse (claims to structured analysis) -> validate -> render.
Six stages, one process, no multi-agent framework. Deterministic Python handles
everything that does not require reasoning.

COST AND EFFICIENCY (measured, not estimated)
- About $0.05 per company report, using gpt-4o-mini for extraction and gpt-4o for
  synthesis once per company
- Repeat runs cost $0.00 through content-addressed caching of both HTTP responses
  and model responses
- 90.8% of filing text is filtered out before any model call: 324,748 characters of
  10-K text reduced to 29,914 characters of targeted context
- Cost ceiling is enforced before the first API call; the run aborts rather than
  silently overspending

SIGNALLING ENGINEERING DETAIL
- Reports evidence coverage per analytical category against advisory targets. When
  a category comes in below target, the shortfall is reported to the reader rather
  than filled with generated content.
- Distinguishes disclosed facts from analytical inference, and labels
  competitor relationships as "disclosed" (named by the company's own filing) or
  "industry" (selected by a reviewer).
- Found and fixed 18 defects through testing against real filings rather than
  synthetic fixtures. The most serious: a year-over-year figure that compared a
  90-day quarter against a 363-day year and reported a confidently wrong number;
  and a table-unit error that produced a figure wrong by a factor of 1000.
- Three integration defects were only discoverable by running against the live API;
  the offline suite passed throughout, which is why live validation is now
  part of the workflow.

DOMAIN ANGLE (differentiating, and true)
- Discovered that Applied Materials' Form 10-K contains the word "deposition" once,
  "etch" once, and names no competitor at all — so competitor analysis built on a
  company's own filing is structurally weak. Solved by ingesting peer-company
  first-party filings for competitive overlap, and by mining 8-K Exhibit 99.1
  earnings releases, where management's actual commentary on AI, HBM and advanced
  packaging demand lives.
- Built an explicit staleness guard after finding that Infineon's most recent annual
  report on EDGAR is from 2009 because it now files only in Europe; the pipeline
  flags such a report as historical rather than presenting it as current.

STACK
Python 3.12, pydantic v2, httpx, Jinja2, PyYAML, pytest. No LangChain, no vector
database, no agent framework — retrieval is lexical and the architecture is
deliberately minimal.
```

---

## Ready-to-use resume bullets

If you would rather paste something directly, these are accurate as written.

**Project entry**

**Semiconductor Company Intelligence Agent** — Python, pydantic, httpx, Jinja2
- Built a solo, config-driven pipeline that turns a company name into a structured
  intelligence report across the chain Technology → Product → Market → Customer →
  Competition → Business, validated on 10 semiconductor companies including Applied
  Materials, Lam Research, ASML, NVIDIA and Micron.
- Designed an evidence gate that requires every model-extracted claim to carry a
  verbatim quote checked against the source filing; tested against 48 deliberately
  fabricated claims, all rejected. Achieved zero dangling citations across 656
  claims by restricting analysis to citing existing claim IDs.
- Eliminated model involvement in numerical data: financials come from SEC XBRL as
  exact values, and segment and geographic revenue are parsed from 10-K MD&A tables
  with deterministic Python, removing the highest-risk hallucination surface.
- Cut per-company analysis cost to about $0.05 by reducing 324,748 characters of
  filing text to 29,914 characters of targeted context (90.8% reduction) and
  caching all HTTP and model responses for free repeat runs.
- Diagnosed that Applied Materials' 10-K names no competitor and mentions
  "deposition" once, then addressed the resulting evidence gap by ingesting peer
  first-party filings and 8-K Exhibit 99.1 earnings releases.
- Wrote 122 offline tests and documented 18 defects found against real filings,
  including a year-over-year metric that silently compared a quarter against a full
  year.

**LinkedIn (2 sentences)**

I built a Python pipeline that generates evidence-grounded intelligence reports on
semiconductor companies, covering technology, products, markets, customers,
competition and financials across 10 companies for roughly five cents per report.
Every qualitative claim is verified against a verbatim quote from a first-party SEC
filing before it can appear, so the analysis is traceable rather than asserted.

---

## Honest caveats (know these before an interview)

Volunteering a limitation reads as seniority; being caught by one reads as
overclaiming. Three to have ready:

1. **Two companies are fully validated end to end** (Applied Materials, Lam
   Research) as configuration-driven additions. The other eight run and produce
   reports, but two of them — KLA and ASML — score poorly on analytical coverage
   because their filings use unusual structures (KLA's 10-K has no item headings in
   the body; ASML's 20-F uses a cross-reference table). This is a known parsing
   limitation, documented in the README.
2. **Infineon produces an empty report** because its most recent annual report on
   EDGAR is from 2009. The pipeline flags this loudly rather than generating
   plausible content — which is the intended behaviour, not a failure.
3. **Cost is about $0.05 per company, not free, and 10 companies cost $0.50.** The
   figure to quote is per-company, and repeat runs are free due to caching.

If asked "what would you do next": fix the 20-F and PART-only section parsing so
KLA and ASML reach full coverage, and add unit-aware table parsing generalisation.
Both are scoped in the README's limitations section.
