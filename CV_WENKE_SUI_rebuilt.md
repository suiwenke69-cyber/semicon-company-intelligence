# WENKE SUI

+65 8741 7700 | Singapore | suiwenke69@gmail.com
github.com/suiwenke69-cyber

**SEMICONDUCTOR TECHNOLOGY · OPERATIONS · COMMERCIAL / PRODUCT · AI**

---

## PROFILE

NUS MSc candidate in Semiconductor Technology and Operations, with a Materials Science
background and hands-on thin-film/process experience. Builds AI tools that turn public
semiconductor and financial data into structured, evidence-traceable analysis — not
demos, but working systems with measured cost, test coverage, and documented
limitations. Combines fab-level process understanding with the ability to map
technology to products, markets, customer requirements and competitive positioning.
Targeting roles at the intersection of technology, product and commercial strategy.

---

## AI & ANALYTICAL TOOLING — SOLO PROJECTS

**Semiconductor Company Intelligence Agent** · Python · *public repo*
*Evidence-grounded analysis pipeline covering the chain Technology → Product → Market → Customer → Competition → Business*
- Built a config-driven pipeline analysed across **10 semiconductor companies** (Applied Materials, Lam Research, ASML, NVIDIA, Micron, KLA, AMD, NXP, Analog Devices, Infineon), generating **656 source-cited claims** from **175 first-party SEC documents**; adding a new company requires editing YAML only, with **zero company-specific code changes**.
- Designed an **anti-hallucination gate**: every model-extracted claim must carry a verbatim quote verified against the source filing in Python. Tested against 48 deliberately fabricated claims — all rejected; achieved **zero dangling citations** across all runs.
- **Removed language models from all numerical work**: revenue, R&D and margins are pulled from SEC XBRL as exact values, and segment/geographic revenue is parsed from 10-K MD&A tables with deterministic Python, eliminating the highest-risk error surface.
- Cut analysis cost to **~$0.05 per company report** by reducing 324,748 characters of filing text to 29,914 characters of targeted context (**90.8% reduction**) and caching all HTTP and model responses so repeat runs cost **$0.00**.
- Diagnosed that Applied Materials' Form 10-K contains the word "deposition" **once** and names **no competitor**, then closed the resulting evidence gap by ingesting peer-company filings and 8-K Exhibit 99.1 earnings releases.
- Wrote **122 offline tests** and documented 18 defects found against real filings, including a year-over-year metric that silently compared a 90-day quarter against a 363-day year, and a table-unit error producing a figure wrong by **1000×**.

**ITrader — Equity Research & Monitoring Assistant** · Python, Streamlit, SQLite
*Personal research tool: technical, fundamental, valuation and monitoring workflow*
- Architected **20,321 lines across 84 modules** on one governing rule — *deterministic code first, LLM only to explain results already computed* — so every conclusion answers "why" with traceable arithmetic.
- Implemented all technical indicators from scratch (MA, RSI, MACD, ATR, RVOL) rather than wrapping a library, plus ATR-filtered swing-point clustering that outputs **support/resistance as ranges** with touch counts, strength scores and per-item rationale instead of single-point levels.
- Built a valuation engine that **auto-selects method by company type** and outputs Bear/Base/Bull ranges with all assumptions exposed; fundamentals extracted from SEC XBRL.
- Designed for **token-cost control as a first principle**: monitoring polls consume **zero tokens** (pure arithmetic module, structurally barred from importing any LLM), routing through three gates — cooldown → market-state hash → daily budget — with `to_llm_payload` whitelisting scalars so **OHLCV series cannot structurally be sent to a model**.
- Delivered **396 offline tests, all passing**, requiring no API keys and incurring no API cost.

**Price Action Academy — Interactive Course Platform** · React, Vite, Supabase, Stripe
- Built a **60-lesson** interactive course (9 modules) with in-browser generated charts, step-through walkthroughs, and **180 quiz questions**; **6,122 lines** of front-end across 46 files.
- Implemented a **commercial stack**, not just content: 12 serverless endpoints handling Stripe checkout and webhooks, phone-based authentication and SMS delivery, entitlements, progress tracking and licensed access, backed by 4 Supabase tables with versioned SQL migrations; deployed on Vercel.

---

## SEMICONDUCTOR INDUSTRY & COMMERCIAL KNOWLEDGE

- **Industry ecosystem**: understands the roles and business relationships among fabless firms, IDMs, foundries, OSATs, equipment/materials suppliers, EDA/IP vendors and distributors.
- **Manufacturing flow**: familiar with wafer fabrication steps including deposition, lithography, etch, ion implantation, CMP and cleaning, followed by packaging and test.
- **Customer perspective**: understands how process requirements, yield, performance, cost, capacity, qualification cycles and technical support shape semiconductor purchasing decisions.
- **Commercial perspective**: able to connect technology to market and customer needs, and to analyse suppliers, customers, process nodes, capex cycles, competitive positioning and industry drivers — demonstrated by the competitive and geographic analysis in the intelligence agent above.

---

## EDUCATION

**National University of Singapore (NUS)** — MSc, Semiconductor Technology and Operations
08/2026 – Present · Focus: semiconductor manufacturing, process integration, operations, industry value chain

**Hokkaido University, Japan** — HUSTEP Exchange Program
09/2025 – 06/2026 · JASSO Scholarship, 05/2025

**Sun Yat-sen University (Project 985)** — BEng, Materials Science and Engineering
09/2022 – 06/2026 · GPA 3.6/4.0 · Xian Weijian International Scholarship, 05/2025

---

## TECHNICAL EXPERIENCE

**Sb₂Se₃ Thin-Film Preparation & Process Optimization** — Sun Yat-sen University · 10/2025 – Present
- Investigated how substrates and magnetron sputtering conditions affect thin-film crystal orientation and morphology, building practical understanding of PVD/deposition process variables.
- Optimised sputtering power, pressure and temperature; evaluated process–structure relationships using XRD, SEM and XPS, and translated characterisation data into process optimisation conclusions.

**Laser Cladding of Titanium Alloy with Transition Layer** — Sun Yat-sen University · 03/2025 – 12/2025
- Adjusted transition-layer composition and compared coating performance across experimental conditions through preparation, testing and data-based comparison.

---

## ADDITIONAL RESEARCH & PUBLICATION

**Co-author**, *Buildings* (SCI Q2) — "A Novel Earth-to-Air Heat Exchanger-Assisted Ventilated Double-Skin Facades for Low-Grade Renewable Energy Utilization in Transparent Building Envelope" · 10/2025
- Supported material characterisation, experimental data acquisition, and validation of numerical simulation against measurements.

**Deep-Learning Recognition of Material Microscopic Features and Defects** · 09/2023 – 09/2024
- Researched algorithms for identifying surface defects in 3D-printed metals and improved validation accuracy through algorithm modification.

---

## SKILLS

**AI / Data**: Python (pydantic, httpx, pandas, numpy, Streamlit) | LLM pipeline design, prompt/cost engineering, deterministic-vs-LLM architecture | pytest (518 tests written across projects) | SQL/SQLite | React, JavaScript | Git
**Semiconductor / Materials**: PVD & magnetron sputtering; thin-film deposition; process optimisation; SEM, XRD, TEM, Raman, UV-Vis; semiconductor fabrication and packaging/test flow
**Commercial**: technical-to-commercial translation | industry and value-chain analysis | competitive and market intelligence | customer-needs orientation
**Languages**: Chinese (Native) | English (IELTS 7.5) | Japanese (Conversational)
