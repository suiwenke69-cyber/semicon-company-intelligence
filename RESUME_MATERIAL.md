# Resume Material: Three AI / Analytical Projects

Everything here is measured from the actual repositories, not estimated. A resume
claim that does not survive a follow-up question is worse than no claim, so every
figure below was verified against the code before being written down.

**Repos**
- Semiconductor Company Intelligence Agent — https://github.com/suiwenke69-cyber/semicon-company-intelligence (public)
- Price Action Academy — https://github.com/suiwenke69-cyber/price-action-academy (private)
- ITrader — local project, not yet on GitHub

**Rebuilt CV** — `CV_WENKE_SUI_rebuilt.html` (open in a browser, then Print → Save as PDF: A4, default margins, background graphics off)

---

## Use this prompt in ChatGPT

Copy the prompt, then paste the context block underneath it.

```
You are helping a graduate student in Semiconductor Technology and Operations
tailor a resume for semiconductor B2B commercial roles: Product Marketing, Market
Intelligence, Business Development, Technical Sales, and Strategic Marketing. The
candidate also wants a clear AI/technical dimension, because they build working AI
tools rather than only studying the industry.

Below is a verified context block covering three solo projects. Rewrite it into:
1. A one-page resume project section: for each project, a title line, a one-line
   subtitle, and 2-4 achievement bullets. Start each bullet with a strong action
   verb and include a concrete number wherever the context provides one.
2. A 2-sentence LinkedIn "Projects" summary covering all three.
3. One STAR-format interview answer (Situation, Task, Action, Result), about 150
   words, for the question "tell me about a technical project you built".

Constraints:
- Use ONLY the numbers and facts in the context block. Do not invent metrics,
  users, revenue, funding, or team size.
- All three are SOLO projects. Do not imply a team or collaborators.
- Where a project produces financial analysis, do not describe it as investment
  advice, a trading system, or revenue-generating. They are research and
  educational tools, and the candidate deliberately avoids predictions and price
  targets.
- Emphasise what matters for the target roles: understanding how a semiconductor
  company makes money, mapping technology to products and markets, customer
  segmentation, competitive positioning, and evidence-based reasoning.
- Keep the tone technical but readable by a non-engineer recruiter.
```

---

## Context block (paste after the prompt)

```
PROJECT 1 — Semiconductor Company Intelligence Agent (Python, public repo)

WHAT IT DOES
Takes a semiconductor company name and produces a structured intelligence report
following Technology -> Product -> Market -> Customer -> Competition -> Business.
Covers 10 companies: Applied Materials, Lam Research, KLA, ASML, Micron, NVIDIA,
AMD, Infineon, NXP, Analog Devices. Adding a new company requires editing one YAML
config file — no Python changes at all.

MEASURED SCALE
- 10 semiconductor companies analysed
- 656 evidence-gated claims extracted
- 175 cited first-party source documents
- 6,838 lines of Python across 21 modules
- 122 offline tests, all passing, zero API cost to run

THREE CORE DESIGN DECISIONS
1. No language model ever touches a number. Revenue, R&D, margins and EPS come from
   SEC XBRL as exact values with accession-level provenance. Segment and geographic
   revenue are parsed from 10-K MD&A tables with deterministic Python table parsing.
2. Anti-hallucination gate: every qualitative claim the model extracts must carry a
   quote copied verbatim from the source filing. The quote is checked against the
   source text in Python; a claim whose quote cannot be found is DISCARDED, not
   softened. On a live test, all 48 deliberately fabricated claims were rejected
   while genuine claims survived.
3. Citation-constrained synthesis: the analysis stage never sees a filing. It
   receives only a claim ledger and may cite only claim IDs that exist, which makes
   citation validation a programmatic check. Result: zero dangling citations across
   all runs.

MEASURED COST AND EFFICIENCY
- About $0.05 per company report (gpt-4o-mini for extraction, gpt-4o once for
  synthesis)
- Repeat runs cost $0.00 via content-addressed caching of HTTP and model responses
- 324,748 characters of 10-K text reduced to 29,914 characters of targeted context
  — a 90.8% reduction — before any model call
- Cost ceiling enforced before the first API call; the run aborts rather than
  silently overspending

DOMAIN INSIGHT (the differentiating part)
- Discovered that Applied Materials' Form 10-K contains the word "deposition" once,
  "etch" once, and names NO competitor at all. Competitor analysis built only on a
  company's own filing is therefore structurally weak. Solved by ingesting
  peer-company first-party filings for competitive overlap, and by mining 8-K
  Exhibit 99.1 earnings releases, where management's actual commentary on AI, HBM
  and advanced packaging demand lives.
- Added a staleness guard after finding that Infineon's most recent annual report on
  EDGAR is from 2009 because it now files only in Europe; the pipeline flags such a
  report as historical rather than presenting it as current intelligence.
- Reports evidence coverage per analytical category against advisory targets, and
  surfaces shortfalls to the reader rather than filling them with generated content.

ENGINEERING DETAIL
- Found and fixed 18 defects by testing against real filings rather than synthetic
  fixtures. Most serious: a year-over-year figure that compared a 90-day quarter
  against a 363-day year and reported a confidently wrong percentage; and a
  table-unit error producing a figure wrong by a factor of 1000.
- Three integration defects were only discoverable by running against the live API;
  the offline test suite passed throughout, which is why live validation is now part
  of the workflow.
- Includes a static HTML front end generated by a CLI flag, with no server and no
  additional dependencies.

STACK
Python 3.12, pydantic v2, httpx, Jinja2, PyYAML, pytest. No LangChain, no vector
database, no multi-agent framework — deliberately minimal.

PROJECT 2 — ITrader: Equity Research & Monitoring Assistant (Python, local)

WHAT IT DOES
A personal US-equity research tool covering four stages of an investor's research
workflow: technical analysis, fundamentals, valuation, and background monitoring
with alerts. It is an educational research aid: it does not predict prices, does not
give price targets, does not auto-trade, and does not connect to a brokerage.

MEASURED SCALE
- 20,321 lines of Python across 84 files
- 396 offline tests, all passing, requiring no API keys and incurring no API cost

CORE ARCHITECTURE RULE
"Deterministic code first, LLM only when reasoning or explanation is needed."
Anything achievable with Python, maths or rules is never sent to a model. The model
only explains results that have already been computed.

KEY IMPLEMENTATION POINTS
- All technical indicators implemented from scratch (MA, RSI, MACD, ATR, RVOL)
  rather than wrapping a third-party library, so behaviour is controllable, testable
  and immune to dependency changes.
- Support and resistance are output as RANGES, not single prices: swing points are
  filtered by ATR significance then clustered, each range carrying touch counts, a
  strength score and a per-item written rationale.
- A valuation engine that automatically selects its method by company type and
  returns Bear / Base / Bull ranges with every assumption exposed. Fundamentals come
  from SEC XBRL filings.
- Token cost treated as a first principle: the monitoring loop consumes ZERO tokens
  because it lives in a pure arithmetic module structurally not allowed to import any
  LLM. A model call can only happen after deterministic rules fire, and only after
  passing three gates — cooldown, market-state hash, and a daily budget.
- The LLM payload is built from an explicit scalar whitelist, so it is structurally
  impossible to send price series (OHLCV) to a model; a test asserts this.
- Multiple market-data providers with rate limiting, plus local caching with TTL and
  delta fetching so only new bars are requested.

STACK
Python 3.12, pandas, numpy, scipy, Streamlit, SQLite, plotly, pytest.

PROJECT 3 — Price Action Academy: Interactive Course Platform (React, private repo)

WHAT IT DOES
An interactive price-action trading course delivered as a web product, with lessons
built from in-browser generated charts, step-by-step walkthroughs, interactive
simulators and quizzes.

MEASURED SCALE
- 60 lessons across 9 modules, all fully written
- 180 quiz questions
- 6,122 lines of front-end code across 46 files
- 12 serverless API endpoints

WHAT MAKES IT A PRODUCT RATHER THAN CONTENT
- Commercial stack implemented end to end: Stripe checkout and webhooks, phone-based
  authentication with SMS delivery, entitlements and licensed access control, and
  progress tracking.
- Backed by 4 Supabase tables (profiles, entitlements, progress, orders) with
  versioned SQL migrations.
- Deployed on Vercel with separate front-end and serverless API layers.
- All course charts and data are generated in-browser, so lesson content has no
  backend dependency.

STACK
React, Vite, JavaScript, Supabase (Postgres), Stripe, serverless functions, Vercel.

CANDIDATE BACKGROUND (for framing the projects)
NUS MSc candidate in Semiconductor Technology and Operations; BEng in Materials
Science (GPA 3.6/4.0) with an exchange year at Hokkaido University; hands-on
thin-film deposition and magnetron sputtering experience with XRD, SEM and XPS;
co-author of an SCI Q2 journal paper. Total across projects: 518 tests written.
Languages: Chinese (native), English (IELTS 7.5), Japanese (conversational).
```

---

## Ready-to-paste resume bullets

**Semiconductor Company Intelligence Agent** — Python, pydantic, httpx, Jinja2
- Built a config-driven pipeline analysed across 10 semiconductor companies, producing 656 source-cited claims from 175 first-party SEC documents; adding a company requires YAML only, with zero company-specific code.
- Designed an anti-hallucination gate requiring every model-extracted claim to carry a verbatim quote verified against the source filing in Python; 48 deliberately fabricated claims were all rejected, with zero dangling citations across every run.
- Removed language models from all numerical work — financials come from SEC XBRL as exact values and segment/geographic revenue is parsed deterministically — eliminating the highest-risk error surface.
- Cut cost to about $0.05 per company report by reducing 324,748 characters of filing text to 29,914 characters of targeted context (90.8%), with caching making repeat runs free.

**ITrader — Equity Research & Monitoring Assistant** — Python, Streamlit, SQLite
- Architected 20,321 lines across 84 modules on one rule — deterministic code first, LLM only to explain results already computed — so every conclusion answers "why" with traceable arithmetic.
- Implemented all technical indicators from scratch and built support/resistance as ATR-filtered ranges with touch counts and strength scores, plus a valuation engine that auto-selects method by company type; 396 offline tests, all passing.
- Made token cost a first principle: monitoring consumes zero tokens because it runs in an arithmetic module structurally barred from importing any LLM, gated by cooldown, state hashing and a daily budget.

**Price Action Academy — Interactive Course Platform** — React, Vite, Supabase, Stripe
- Built a 60-lesson interactive course across 9 modules with in-browser generated charts and 180 quiz questions; 6,122 lines of front-end across 46 files.
- Implemented a commercial stack, not just content: 12 serverless endpoints for Stripe checkout and webhooks, phone authentication with SMS, entitlements and licensed access, backed by 4 Supabase tables with versioned SQL migrations; deployed on Vercel.

---

## One thing to fix before sharing

**Price Action Academy's README understates the project.** It currently says:

> Lessons 01–21 are fully written (charts, step walkthroughs, quizzes). Lessons 22–60
> are catalogued placeholders.

That is stale. Inspection of `api/_content/lessons.json` shows **all 60 lessons carry
full content** — concept, narrative, bull/bear framing, common mistake, rule, figures,
interactive prompt and quiz — across 9 modules. A recruiter who opens that repo would
conclude the project is roughly one-third complete when it is finished.

Worth updating that README before sending the link to anyone.

---

## Honest caveats (know these before an interview)

Volunteering a limitation reads as seniority; being caught by one reads as
overclaiming. Four to have ready:

1. **Only two companies are fully validated** as configuration-driven additions
   (Applied Materials, Lam Research). The other eight produce reports, but KLA and
   ASML score poorly on analytical coverage because their filings use unusual
   structures — KLA's 10-K has no item headings in the body, and ASML's 20-F uses a
   cross-reference table. Documented in the project README.
2. **Infineon produces an empty report** because its most recent annual report on
   EDGAR is from 2009; the pipeline flags this loudly instead of generating plausible
   content. That is intended behaviour, not a failure.
3. **ITrader and Price Action Academy have no users.** They are solo builds, not
   products with traction. Describe them as working systems demonstrating engineering
   and product thinking, not as businesses.
4. **Cost is about $0.05 per company, not free.** Quote the per-company figure and note
   that repeat runs are free because of caching.

If asked "what would you do next": generalise the 20-F and PART-only section parsing
so KLA and ASML reach full coverage, automate table-unit detection across all filers,
and add per-sector coverage targets instead of one global set.
