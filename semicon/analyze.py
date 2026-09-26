"""Stage 4: synthesise the structured analysis from the claim ledger.

The contract this stage depends on, and the reason the pipeline is auditable:

    analysis consumes Evidence records and NOTHING ELSE

It never sees a filing. Every assertion it makes must cite ``claim_id`` values
that exist in the ledger. That constraint is enforced afterwards by
``validate.py``, which means a fabricated product, customer or competitor is
detectable by a Python assertion rather than by a human re-reading the 10-K.

The model's job here is genuinely different from extraction: classification,
comparison and reasoning rather than transcription. That is why synthesis runs on
the stronger model while extraction runs on the cheap one.
"""

from __future__ import annotations

import json
import re

from .config import Settings
from .llm.client import LlmClient, LlmError
from .models import (
    BusinessSegment,
    ClaimType,
    CommercialInsight,
    Competitor,
    Confidence,
    CustomerDisclosure,
    CustomerInsight,
    Evidence,
    FinancialMetric,
    GeographicRevenue,
    MarketDriver,
    ProcessStep,
    ProductItem,
    RevenueBreakdown,
    RiskItem,
    TechnologyItem,
)

SYSTEM_PROMPT = """You are a semiconductor industry analyst writing the analytical sections of a company intelligence report for a B2B audience (product marketing, market intelligence, business development, technical sales).

You will receive a numbered ledger of CLAIMS, each taken from SEC filings and each already verified to have a verbatim source. You have no other information about the company, and you must not use any.

Hard rules:
1. Every item you output must include "sources": a list of claim ids copied exactly from the ledger (e.g. ["c_1a2b3c4d5e"]). Use only ids present in the ledger.
2. Never state a fact that the ledger does not contain. If the ledger does not cover something, leave that field empty rather than filling it with what you know about the industry.
3. Do NOT name a specific customer unless a claim names it AND says it is a customer. Otherwise use a customer CATEGORY (for example "leading-edge logic foundries") and set disclosure to "INFERRED_CATEGORY" with a "reasoning" field explaining the basis.
4. For process steps, choose ONLY from this closed list: {process_steps}. Do not invent steps.
5. Avoid subjective rankings. Do not call a company or product "best", "leading" or "dominant" unless a claim quotes the filing saying so. Describe positioning in terms of what the products do and where they overlap.

SECTION DENSITY - important:
The ledger usually supports more entries than a first pass produces. Do not collapse distinct evidence into one generic entry. Specifically:
- customers: create a SEPARATE entry for each distinct customer category the ledger supports. A filing that names foundry, logic, DRAM and NAND customers supports four entries, not one "manufacturers of semiconductors" entry. Each entry should carry its own requirements and pain points.
- markets: create a separate entry for each distinct demand driver (AI infrastructure, HBM, advanced packaging, leading-edge logic, memory cycles, foundry capex). Do not merge them into one.
- commercial_insights: aim for 5-8 distinct insights drawn from different evidence. One insight is almost always an under-use of the ledger.
- competitors: one entry per named competitor.
- technologies: one entry per distinct technology or capability.

Covering more of the ledger is the goal, but every entry must still cite supporting claim ids. Density is achieved by using MORE of the evidence, never by inventing entries the ledger does not support. If the ledger only supports one customer category, output one.

Distinguish fact from inference:
- Set claim_type to "FACT" when the claim states it directly.
- Set claim_type to "ANALYSIS" when you are reasoning across two or more claims.
- Set claim_type to "INFERENCE" when you are reasoning with an explicit chain, and state that chain in "reasoning".

Return JSON only, matching this shape exactly (omit nothing, use empty lists where you have no support):
{schema}"""


def _process_step_list(configured: list[str]) -> list[str]:
    """The closed process-step taxonomy offered to the model.

    Falls back to the full enum when a company has no configured steps, so a
    newly added company still gets a bounded choice rather than free rein.
    """
    if configured:
        return configured
    return [p.value for p in ProcessStep]


def _schema_for_output(process_steps: list[str]) -> dict:
    """The JSON schema shown to the model, built from the closed taxonomy.

    The process-step enum is injected rather than hard-coded so that adding a
    company with different process steps needs no prompt edit.
    """
    return {
        "value_chain_position": "string - where the company sits in the semiconductor value chain",
        "business_segments": [
            {
                "name": "string",
                "description": "string",
                "products": ["string"],
                "sources": ["claim_id"],
            }
        ],
        "products": [
            {
                "name": "string",
                "category": "string",
                "description": "string",
                "process_steps": [f"one of: {process_steps}"],
                "technologies": ["string"],
                "sources": ["claim_id"],
            }
        ],
        "technologies": [
            {
                "name": "string",
                "description": "string",
                "differentiator": "string or null",
                "roadmap_note": "string or null",
                "sources": ["claim_id"],
            }
        ],
        "markets": [
            {
                "driver": "string - e.g. AI infrastructure, HBM, advanced packaging",
                "relevance": "string - why this applies to THIS company",
                "company_exposure": "string or null",
                "claim_type": "FACT | ANALYSIS | INFERENCE",
                "sources": ["claim_id"],
            }
        ],
        "customers": [
            {
                "name": "string - a named customer, or a customer category",
                "disclosure": "DISCLOSED | INFERRED_CATEGORY",
                "category": "string or null",
                "requirements": ["string"],
                "pain_points": ["string"],
                "value_created": "string",
                "reasoning": "string - required when disclosure is INFERRED_CATEGORY",
                "sources": ["claim_id"],
            }
        ],
        "competitors": [
            {
                "name": "string",
                "basis": "disclosed | industry - use 'disclosed' ONLY when the subject company's own filing names this peer; otherwise 'industry'",
                "product_overlap": "string",
                "technology_positioning": "string",
                "market_exposure": "string",
                "strengths": ["string"],
                "potential_weaknesses": ["string"],
                "sources": ["claim_id"],
            }
        ],
        "growth_drivers": [
            {
                "driver": "string",
                "relevance": "string",
                "claim_type": "FACT | ANALYSIS | INFERENCE",
                "sources": ["claim_id"],
            }
        ],
        "risks": [
            {
                "risk": "string",
                "category": "string - e.g. cyclical, competitive, customer concentration, regulatory, supply chain",
                "claim_type": "FACT | ANALYSIS | INFERENCE",
                "mitigation_note": "string or null",
                "sources": ["claim_id"],
            }
        ],
        "commercial_insights": [
            {
                "insight": "string",
                "kind": "growth_opportunity | technology_opportunity | customer_opportunity | competitive_threat | business_risk",
                "claim_type": "ANALYSIS | INFERENCE",
                "reasoning": "string",
                "sources": ["claim_id"],
            }
        ],
    }


def build_ledger_text(
    claims: list[Evidence],
    *,
    max_claims: int = 160,
    max_excerpt_chars: int = 90,
) -> str:
    """Render the claim ledger for the prompt.

    Claims are grouped by topic and truncated, because the point of this stage is
    reasoning over a compact ledger rather than re-reading prose. The full ledger
    is preserved in the profile regardless.
    """
    grouped: dict[str, list[Evidence]] = {}
    for c in claims:
        grouped.setdefault(c.topic or "company_overview", []).append(c)

    lines: list[str] = []
    used = 0
    for topic in sorted(grouped, key=lambda t: -len(grouped[t])):
        lines.append(f"\n## {topic}")
        for c in grouped[topic]:
            if used >= max_claims:
                break
            excerpt = (c.excerpt or "")[:max_excerpt_chars].replace("\n", " ")
            lines.append(
                f"[{c.claim_id}] ({c.source_id}) {c.text}"
                + (f"\n    quote: \"{excerpt}\"" if excerpt else "")
            )
            used += 1
        if used >= max_claims:
            lines.append("\n(ledger truncated for length)")
            break
    return "\n".join(lines)


def _coerce_process_steps(values: list[str], allowed: list[str]) -> list[ProcessStep]:
    """Map free-text process-step labels onto the closed enum.

    Anything that does not map is dropped rather than coerced, so an invented
    process step cannot reach the report.

    Compound values are split first. A model asked for process steps commonly
    answers "deposition and etch" or "Deposition, Etch, CMP" inside a single list
    element, and matching the whole phrase resolves nothing - which is how a ledger
    containing four process-step claims produced an empty process-step section.
    """
    valid = {p.value.lower(): p for p in ProcessStep}
    out: list[ProcessStep] = []
    allowed_lower = {a.lower() for a in allowed}

    def match_one(label: str) -> ProcessStep | None:
        label = label.strip().lower()
        if not label:
            return None
        found = valid.get(label)
        if found is None:
            for key, enum_val in valid.items():
                if key in label or label in key:
                    found = enum_val
                    break
        if found is None:
            return None
        # Respect the company's configured scope when one is set.
        if allowed_lower and found.value.lower() not in allowed_lower:
            return None
        return found

    for raw in values or []:
        text = (raw or "").strip()
        if not text:
            continue
        for part in re.split(r"[,/;&]|\band\b", text, flags=re.IGNORECASE):
            found = match_one(part)
            if found is None:
                # Fall back to the unsplit phrase, in case a legitimate label
                # itself contains one of the split tokens.
                found = match_one(text)
            if found is not None and found not in out:
                out.append(found)
    return out


def process_steps_from_claims(
    claims: list[Evidence], allowed: list[str]
) -> list[ProcessStep]:
    """Derive process steps from the claim ledger as a fallback.

    Products are not the only place process steps appear: a filing may describe a
    process step in its business narrative without the synthesis stage attaching it
    to a specific product. Scanning the process-step claims recovers those, so the
    reported list reflects the evidence rather than only the subset that happened to
    be mapped onto products.
    """
    discovered: list[ProcessStep] = []
    for claim in claims:
        if (claim.topic or "").lower() != "process_step":
            continue
        for step in _coerce_process_steps([claim.text], allowed):
            if step not in discovered:
                discovered.append(step)
    return discovered


def _coerce_evidence(
    values: list, model_cls, allowed_ids: set[str]
) -> tuple[list, int]:
    """Validate a list of model-generated objects and strip unknown citations.

    Unknown claim ids are removed rather than kept, so a citation that cannot be
    resolved never reaches the report. Items left with no citation at all are
    dropped, because an uncited assertion is exactly what this pipeline exists to
    prevent.
    """
    out = []
    dropped = 0
    for item in values or []:
        try:
            obj = model_cls.model_validate(item)
        except Exception:
            dropped += 1
            continue
        obj.sources = [s for s in (obj.sources or []) if s in allowed_ids]
        if not obj.sources:
            dropped += 1
            continue
        out.append(obj)
    return out, dropped


def analyse(
    claims: list[Evidence],
    *,
    client: LlmClient,
    settings: Settings,
    company_name: str,
    process_steps: list[str] | None = None,
    financials: list[FinancialMetric] | None = None,
    segments: list[BusinessSegment] | None = None,
    geography: list[GeographicRevenue] | None = None,
    breakdowns: list[RevenueBreakdown] | None = None,
    peer_evidence: list[Evidence] | None = None,
) -> tuple[dict, list[str]]:
    """Produce the analytical sections, plus warnings.

    Returns a plain dict that the pipeline assembles into the profile. Keeping it
    a dict here means this stage has no dependency on how the profile is finally
    shaped.
    """
    allowed = _process_step_list(process_steps or [])
    schema = _schema_for_output(allowed)
    system = SYSTEM_PROMPT.replace("{process_steps}", ", ".join(allowed)).replace(
        "{schema}", json.dumps(schema, indent=2)
    )

    ledger = build_ledger_text(claims)
    warnings: list[str] = []

    # Deterministic facts are supplied as context so the model's analysis is
    # grounded in the real numbers, but it is told these are not its to invent.
    context_blocks: list[str] = []
    if financials:
        rows = []
        for m in financials:
            period = m.latest_annual or m.latest
            if period and period.value is not None:
                rows.append(
                    f"{m.metric}: {period.value:,.0f} {m.unit} "
                    f"({period.period_label}"
                    + (f", {m.yoy_growth_pct:+.1f}% YoY" if m.yoy_growth_pct is not None else "")
                    + ")"
                )
        if rows:
            context_blocks.append(
                "## Verified financial figures (from XBRL, already exact - cite as "
                "'sec_xbrl_facts', do not restate as claims)\n" + "\n".join(rows)
            )
    if segments:
        context_blocks.append(
            "## Verified segment revenue (parsed from the 10-K MD&A)\n"
            + "\n".join(
                f"{s.name}: {s.revenue_millions:,.0f}M "
                f"({s.revenue_pct_of_total:.0f}% of total)"
                for s in segments
                if s.revenue_millions
            )
        )
    if geography:
        context_blocks.append(
            "## Verified geographic revenue (parsed from the 10-K MD&A)\n"
            + "\n".join(
                f"{g.region}: {g.revenue_millions:,.0f}M "
                f"({g.pct_of_total:.0f}%)"
                + (" [subtotal]" if g.is_subtotal else "")
                for g in geography
                if g.revenue_millions
            )
        )

    # Peer-filing evidence. This is where a competitor section gets real content:
    # the subject's own 10-K often names nobody, while a peer's filing states
    # exactly what that peer makes. Kept clearly separated so the model knows this
    # text describes ANOTHER company.
    if peer_evidence:
        peer_lines = []
        for c in peer_evidence[:60]:
            peer_lines.append(f"[{c.claim_id}] (from {c.source_id}) {c.text}")
        context_blocks.append(
            "## Competitive-overlap evidence from PEER filings\n"
            "These statements describe OTHER companies, taken from their own SEC "
            "filings. Use them to assess product and technology overlap. They are "
            "typed INFERENCE: never present them as facts about "
            f"{company_name}, and cite their claim ids.\n"
            + "\n".join(peer_lines)
        )

    user = (
        f"Company: {company_name}\n"
        + ("\n\n".join(context_blocks) + "\n\n" if context_blocks else "")
        + "--- CLAIM LEDGER ---\n"
        + ledger
        + "\n--- END LEDGER ---\n\n"
        "Produce the analysis JSON. Cite only the claim ids shown above."
    )

    try:
        payload = client.complete_json(
            model=settings.llm.synth_model,
            system=system,
            user=user,
            max_tokens=settings.llm.max_output_tokens,
        )
    except LlmError as exc:
        warnings.append(f"synthesis failed: {exc}")
        return {}, warnings

    if not isinstance(payload, dict):
        warnings.append("synthesis returned a non-object payload")
        return {}, warnings

    # Peer claims are citable for the competitor section, but they remain in their
    # own ledger. Allowing the ids here is what lets the model ground competitive
    # overlap in a real document instead of asserting it.
    allowed_ids = {c.claim_id for c in claims} | {
        c.claim_id for c in (peer_evidence or [])
    }

    # Validate each section independently so one malformed list does not discard
    # the whole analysis.
    result: dict = {}
    for key, model_cls in (
        ("business_segments", BusinessSegment),
        ("products", ProductItem),
        ("technologies", TechnologyItem),
        ("markets", MarketDriver),
        ("customers", CustomerInsight),
        ("competitors", Competitor),
        ("growth_drivers", MarketDriver),
        ("risks", RiskItem),
        ("commercial_insights", CommercialInsight),
    ):
        items, dropped = _coerce_evidence(payload.get(key) or [], model_cls, allowed_ids)
        if dropped:
            warnings.append(f"synthesis: dropped {dropped} unsupported {key} item(s)")
        result[key] = items

    # Normalise process steps onto the closed taxonomy.
    #
    # Note: pydantic has already coerced these values into ProcessStep members, so
    # they must be unwrapped via .value. Using str() here produced
    # 'ProcessStep.ADVANCED_PACKAGING', which matches nothing in the taxonomy - so
    # the report claimed no process steps were identified while the ledger plainly
    # contained four of them.
    for product in result.get("products", []):
        product.process_steps = _coerce_process_steps(
            [
                s.value if isinstance(s, ProcessStep) else str(s)
                for s in product.process_steps
            ],
            allowed,
        )

    # Enforce the customer-disclosure rule from the specification.
    for customer in result.get("customers", []):
        if customer.disclosure == CustomerDisclosure.INFERRED_CATEGORY and not customer.reasoning:
            customer.reasoning = (
                "Category inferred from the company's disclosed end markets; no "
                "specific customer is named in the cited filing."
            )

    if isinstance(payload.get("value_chain_position"), str):
        result["value_chain_position"] = payload["value_chain_position"]

    return result, warnings
