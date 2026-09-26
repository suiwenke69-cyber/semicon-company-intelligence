"""Stage 3: extract source-grounded claims from selected filing text.

This is the first stage that spends money, so it is designed around a single
principle: **the model may only report what it can quote.**

Every claim must carry a verbatim quote. Claims without one are discarded in
Python rather than being trusted, which means a hallucinated product or customer
has to survive quoting text that does not exist. This is what the specification
asked for when it warned about invented semiconductor information; it is enforced
here, not merely requested in a prompt.

One model call per chunk, deliberately. Batching several chunks per call would
save a little prompt overhead but would make a bad response much harder to
attribute to a source, and would risk losing all claims in a chunk when one
fails.
"""

from __future__ import annotations

import hashlib
import re

from .chunk import Chunk
from .config import Settings
from .llm.client import LlmClient, LlmError
from .models import (
    EXTRACTION_TOPICS,
    ClaimType,
    Confidence,
    Evidence,
    ExtractedClaim,
    ExtractionResult,
)

SYSTEM_PROMPT = """You extract factual claims from SEC filings about semiconductor companies.

You are building an evidence ledger for a market-intelligence report. Another analyst will use ONLY your output, so anything you cannot support with the filing's own words is worse than useless - it is a liability.

Absolute requirements:
1. Every claim MUST include a "quote" copied verbatim, word for word, from the provided text. Do not paraphrase, shorten, correct or merge quotes.
2. If you cannot find a verbatim quote supporting a statement, do not emit that statement.
3. Do NOT use outside knowledge. If the text does not say it, you do not know it.
4. Do NOT name specific customers or competitors unless the text names them.

What to extract, treating these as equally important:
- Products, and what they do
- Technologies and technical differentiators
- Process steps (deposition, etch, metrology, and so on)
- End markets and demand drivers
- Named competitors, and how the text frames competition
- Named or clearly described customer categories, plus customer requirements and pain points
- COMPANY-SPECIFIC RISKS. A Risk Factors section is a primary source, not boilerplate. Extract a risk whenever it names something concrete about THIS company: customer concentration, geographic concentration, export controls, supply-chain dependency, litigation, intellectual-property exposure, cyclicality of its specific end markets, or reliance on a particular technology.
- Segment performance, backlog, and management commentary on results

Skip only genuinely content-free material: legal disclaimers, forward-looking-statement notices, page numbers, table-of-contents entries, and risk language that would read identically for every company in the industry.

Write each "claim" as ONE clear factual sentence. Prefer specific and quantitative statements over vague ones ("net revenue in China fell 16% to $8,529 million" beats "China results declined").

Choose exactly one "topic" from this list:
company_overview, product, technology, process_step, customer, competitor, market_driver, financial_commentary, risk, strategy, geography

"confidence" is high when the text states it plainly, medium when it requires mild reading, low when it is implied.

Return JSON only, in this exact shape:
{"claims": [{"claim": "...", "quote": "...", "topic": "...", "confidence": "high"}]}

Return at most 20 claims. Fewer, well-chosen claims beat many weak ones. If the text contains nothing worth extracting, return {"claims": []}."""

# Topics that matter most, used to prioritise which chunks get processed when a
# run is capped. Order is not a ranking of importance to the reader, but of
# extraction value per token spent.
_TOPIC_WEIGHT = {
    "product": 3,
    "process_step": 3,
    "technology": 3,
    "customer": 2,
    "competitor": 2,
    "market_driver": 2,
    "risk": 2,
    "strategy": 1,
    "company_overview": 1,
    "financial_commentary": 1,
    "geography": 1,
}


def _build_user_prompt(
    chunk: Chunk,
    company_name: str,
    process_steps: list[str],
    coverage_hint: str | None = None,
) -> str:
    """Assemble the per-chunk prompt.

    ``coverage_hint`` is how the pipeline closes evidence gaps without lowering the
    evidence bar: it names the categories that are still thin and asks the model to
    look for them specifically. The quote gate is unchanged, so a hint can only
    redirect attention - it cannot cause an unsupported claim to survive.
    """
    steps = ", ".join(process_steps) if process_steps else "(none specified)"
    focus = f"\n{coverage_hint}\n" if coverage_hint else ""
    return f"""Company: {company_name}
Filing section: {chunk.locator}

If a passage describes a semiconductor process step, classify it using ONLY these labels: {steps}
{focus}
--- BEGIN FILING TEXT ---
{chunk.text}
--- END FILING TEXT ---

Extract the factual claims. Remember: every claim needs a verbatim quote from the text above, and choose one topic from the allowed list."""


def _normalise_quote(text: str) -> str:
    """Collapse whitespace so verbatim quotes survive line-break differences."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _verify_quote(chunk_text: str, quote: str | None) -> tuple[bool, str]:
    """Check that a quote actually appears in the chunk.

    This is the anti-hallucination gate, and it is the reason stage 3 output can
    be trusted further downstream. Whitespace and case are normalised, because
    filings break lines arbitrarily inside sentences; nothing else is, so a
    reworded or invented quote is rejected.
    """
    if not quote or len(quote.strip()) < 12:
        return False, "quote missing or too short to verify"
    haystack = _normalise_quote(chunk_text)
    needle = _normalise_quote(quote)
    if needle in haystack:
        return True, ""
    # Allow a long quote to match on its first clause: models sometimes append
    # the start of the next sentence. Still requires a substantial verbatim run.
    if len(needle) > 60:
        head = needle[: int(len(needle) * 0.7)].rstrip()
        if head in haystack:
            return True, ""
    return False, "quote not found verbatim in source text"


def _claim_id(source_id: str, chunk_id: str, index: int, text: str) -> str:
    """Deterministic claim identifier.

    Deterministic rather than sequential so that re-running extraction over
    unchanged text yields identical ids. That keeps citations in a previously
    generated profile valid, and makes profile diffs meaningful.
    """
    digest = hashlib.sha256(f"{chunk_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return f"c_{digest[:10]}"


def extract_chunk(
    chunk: Chunk,
    *,
    client: LlmClient,
    settings: Settings,
    company_name: str,
    process_steps: list[str] | None = None,
    coverage_hint: str | None = None,
) -> ExtractionResult:
    """Extract claims from one chunk, with quote verification."""
    result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        locator=chunk.locator,
    )

    try:
        payload = client.complete_json(
            model=settings.llm.extract_model,
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(
                chunk, company_name, process_steps or [], coverage_hint
            ),
        )
    except LlmError as exc:
        result.warnings.append(f"extraction failed: {exc}")
        return result

    raw_claims = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(raw_claims, list):
        result.warnings.append("model returned no usable 'claims' array")
        return result

    max_excerpt = settings.report.excerpt_max_chars

    for i, item in enumerate(raw_claims):
        try:
            parsed = ExtractedClaim.model_validate(item)
        except Exception:
            result.dropped_invalid += 1
            continue

        ok, reason = _verify_quote(chunk.text, parsed.quote)
        if not ok:
            # The central gate. An unquotable claim is discarded, not softened.
            result.dropped_invalid += 1
            result.warnings.append(f"dropped unverifiable claim ({reason})")
            continue

        topic = (parsed.topic or "").strip().lower()
        if topic not in EXTRACTION_TOPICS:
            topic = "company_overview"

        claim_text = re.sub(r"\s+", " ", parsed.claim).strip()
        if len(claim_text) < 15:
            result.dropped_invalid += 1
            continue

        result.claims.append(
            Evidence(
                claim_id=_claim_id(chunk.source_id, chunk.chunk_id, i, claim_text),
                text=claim_text,
                source_id=chunk.source_id,
                locator=chunk.locator,
                excerpt=(parsed.quote or "")[:max_excerpt],
                confidence=parsed.confidence,
                topic=topic,
            )
        )

    return result


def extract_evidence(
    chunks: list[Chunk],
    *,
    client: LlmClient,
    settings: Settings,
    company_name: str,
    process_steps: list[str] | None = None,
    progress: object | None = None,
    coverage: bool = True,
    narrow_chunks: list[Chunk] | None = None,
) -> tuple[list[Evidence], list[str]]:
    """Run extraction over the selected chunks and return a deduplicated ledger.

    Two passes:

    1. **Broad.** Every selected chunk is extracted once, with no steering. This
       finds whatever the sources naturally yield.
    2. **Targeted.** Coverage is measured. If a category is thin or absent, a
       second pass revisits the highest-value chunks with a hint naming the missing
       categories.

    The second pass exists because one unsteered pass produced heavily skewed
    evidence for Applied Materials: 15 market-driver claims against 2 product
    claims, which made the report read as a news summary rather than product
    intelligence. Critically it does NOT relax the evidence bar: every claim from
    every pass must carry a verbatim quote, so a hint can only redirect attention.
    If the sources genuinely lack product detail, the category stays short and that
    shortfall is reported rather than filled.

    Returns ``(claims, warnings)``. Failures on individual chunks are collected as
    warnings and never abort the run: a report built from 12 of 16 chunks is
    still useful, provided the gap is visible.
    """
    all_claims: list[Evidence] = []
    warnings: list[str] = []
    seen_text: set[str] = set()

    def harvest(outcome: ExtractionResult, label: str) -> None:
        warnings.extend(f"{label}: {w}" for w in outcome.warnings)
        if outcome.dropped_invalid:
            warnings.append(
                f"{label}: {outcome.dropped_invalid} claim(s) discarded for "
                f"lacking a verifiable quote"
            )
        for claim in outcome.claims:
            # Cross-chunk duplicate removal: overlapping sections repeat claims.
            key = _normalise_quote(claim.text)
            if key in seen_text:
                continue
            seen_text.add(key)
            all_claims.append(claim)

    # ---- Pass 1: broad ---------------------------------------------------- #
    for n, chunk in enumerate(chunks, 1):
        if callable(progress):
            progress(n, len(chunks), chunk)
        try:
            outcome = extract_chunk(
                chunk,
                client=client,
                settings=settings,
                company_name=company_name,
                process_steps=process_steps,
            )
        except LlmError as exc:
            warnings.append(f"{chunk.chunk_id}: {exc}")
            continue
        harvest(outcome, chunk.chunk_id)

    if not coverage:
        return all_claims, warnings

    # ---- Pass 2: targeted at measured gaps -------------------------------- #
    from .coverage import coverage_hint, measure_coverage

    report = measure_coverage(all_claims)
    gaps = report.gaps_for_hint()
    if not gaps:
        return all_claims, warnings

    targets = narrow_chunks if narrow_chunks is not None else chunks
    if not targets:
        return all_claims, warnings

    warnings.append(
        "coverage pass: after pass 1 these categories were below target "
        f"({', '.join(gaps)}); a targeted second pass was run over "
        f"{len(targets)} chunk(s)"
    )

    for n, chunk in enumerate(targets, 1):
        if callable(progress):
            progress(n, len(targets), chunk)

        # Re-measure each time: once a category reaches target there is no reason
        # to keep asking for it, and continuing to ask invites padding.
        current = measure_coverage(all_claims)
        if not current.gaps_for_hint():
            warnings.append(
                f"coverage pass: all targets met after {n - 1} chunk(s); stopped early"
            )
            break

        try:
            outcome = extract_chunk(
                chunk,
                client=client,
                settings=settings,
                company_name=company_name,
                process_steps=process_steps,
                coverage_hint=coverage_hint(current, process_steps),
            )
        except LlmError as exc:
            warnings.append(f"{chunk.chunk_id} (coverage pass): {exc}")
            continue
        harvest(outcome, f"{chunk.chunk_id} (coverage)")

    return all_claims, warnings


def claims_by_topic(claims: list[Evidence]) -> dict[str, list[Evidence]]:
    """Group the ledger by topic, preserving order within each topic."""
    grouped: dict[str, list[Evidence]] = {}
    for c in claims:
        grouped.setdefault(c.topic or "company_overview", []).append(c)
    return grouped


def topic_weight(topic: str | None) -> int:
    return _TOPIC_WEIGHT.get(topic or "", 1)

PEER_SYSTEM_PROMPT = """You extract competitive-overlap evidence from a COMPETITOR's SEC filing, for use in a report about a different company.

Context that matters: a company rarely names its competitors in its own filings. AMAT's Form 10-K names none. Peer filings are therefore the only first-party source for what a competitor actually makes - which is what makes technology overlap checkable instead of asserted.

Absolute requirements:
1. Every claim MUST include a "quote" copied verbatim from the provided competitor text.
2. Describe the COMPETITOR's products, process steps and technologies - do not assert anything about the subject company from this text alone.
3. Do NOT claim that the subject company competes with this peer in a specific market unless the competitor text itself states such competition. Overlap is assessed later, by a human-facing analysis step, from what this peer makes.
4. Do NOT use outside knowledge.

Write each "claim" as ONE sentence describing what the competitor offers or does, e.g. "Lam Research supplies dielectric etch systems for conductor and dielectric patterning."

Choose one "topic" from: product, technology, process_step, market_driver, customer, strategy

Return JSON only:
{"claims": [{"claim": "...", "quote": "...", "topic": "...", "confidence": "high"}]}

Return at most 12 claims. If the passage contains no product or technology detail, return {"claims": []}."""


def extract_peer_evidence(
    chunks: list[Chunk],
    *,
    client: LlmClient,
    settings: Settings,
    subject_name: str,
    progress: object | None = None,
) -> tuple[list[Evidence], list[str]]:
    """Extract competitive-overlap evidence from peer filings.

    Peer evidence is deliberately kept OUT of the subject company's ledger. Mixing
    them would inflate the subject's claim count with statements about other
    companies, and would blur the distinction the report depends on: what the
    subject says about itself versus what a peer's disclosure implies about overlap.

    Claims are typed INFERENCE and carry the peer's filing as their source, so a
    reader can always see whose document a statement came from.
    """
    claims: list[Evidence] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for n, chunk in enumerate(chunks, 1):
        if callable(progress):
            progress(n, len(chunks), chunk)
        try:
            payload = client.complete_json(
                model=settings.llm.extract_model,
                system=PEER_SYSTEM_PROMPT,
                user=_build_user_prompt(chunk, subject_name, []),
            )
        except LlmError as exc:
            warnings.append(f"peer {chunk.chunk_id}: {exc}")
            continue

        raw = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            continue

        for i, item in enumerate(raw):
            try:
                parsed = ExtractedClaim.model_validate(item)
            except Exception:
                continue
            ok, reason = _verify_quote(chunk.text, parsed.quote)
            if not ok:
                warnings.append(
                    f"peer {chunk.chunk_id}: dropped unverifiable claim ({reason})"
                )
                continue
            text = re.sub(r"\s+", " ", parsed.claim).strip()
            if len(text) < 15 or _normalise_quote(text) in seen:
                continue
            seen.add(_normalise_quote(text))
            topic = (parsed.topic or "").strip().lower()
            if topic not in EXTRACTION_TOPICS:
                topic = "product"
            claims.append(
                Evidence(
                    claim_id=_claim_id("peer", chunk.chunk_id, i, text),
                    text=text,
                    # A peer's self-description is not a fact about the subject.
                    claim_type=ClaimType.INFERENCE,
                    source_id=chunk.source_id,
                    locator=chunk.locator,
                    excerpt=(parsed.quote or "")[: settings.report.excerpt_max_chars],
                    confidence=parsed.confidence,
                    topic=topic,
                )
            )

    return claims, warnings
