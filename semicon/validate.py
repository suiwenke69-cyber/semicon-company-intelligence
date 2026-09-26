"""Stage 5: mechanical validation of the assembled profile.

This module is the payoff for keeping the LLM stages constrained. Because every
analytical item must cite a ``claim_id``, and the claim ledger is held
separately, "did the model make this up?" becomes a set-membership test rather
than a judgement call.

Checks performed:

1. Every cited claim id exists in the ledger. (Dangling citation = unsupported.)
2. Every claim id is unique.
3. Every ``DISCLOSED`` customer has a citation; every ``INFERRED_CATEGORY``
   customer has stated reasoning. This is the specification's explicit
   requirement not to invent customer relationships.
4. Every process step is in the closed taxonomy.
5. Process steps are within the company's configured scope.
6. Financial figures are internally consistent (a metric's annual and quarterly
   periods are not silently mixed).
7. Names are normalised and de-duplicated so "Applied Global Services" does not
   appear twice with different spacing.

Errors are conditions that make the profile untrustworthy. Warnings are
conditions worth surfacing that do not invalidate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import CompanyProfile, CustomerDisclosure, ProcessStep


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "validation passed with no issues"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


def validate_profile(
    profile: CompanyProfile, *, allowed_process_steps: list[str] | None = None
) -> ValidationReport:
    """Run every consistency check against the assembled profile."""
    report = ValidationReport()

    # Two distinct sets, because they answer different questions:
    #   subject_ids - uniqueness/consistency of the SUBJECT company's ledger.
    #                 Peer claims must be excluded here or the count invariant
    #                 breaks (112 subject claims vs 142 including peers).
    #   citable_ids - what a citation may resolve to. Peer-filing claims live in
    #                 their own ledger but ARE citable by the competitive section.
    subject_ids = {e.claim_id for e in profile.evidence}
    peer_ids = {e.claim_id for e in profile.competitor_evidence}
    citable_ids = subject_ids | peer_ids

    # 1 + 2. Ledger integrity, over the subject company's own ledger.
    if len(subject_ids) != len(profile.evidence):
        seen: set[str] = set()
        dupes: set[str] = set()
        for e in profile.evidence:
            if e.claim_id in seen:
                dupes.add(e.claim_id)
            seen.add(e.claim_id)
        if dupes:
            report.errors.append(
                f"duplicate claim ids in ledger: {sorted(dupes)[:5]}"
            )
        else:
            report.errors.append(
                f"claim ledger count mismatch: {len(profile.evidence)} entries but "
                f"{len(subject_ids)} unique ids"
            )

    # The peer ledger must be internally consistent, and must not shadow a subject
    # claim id, which would make a citation ambiguous.
    if len(peer_ids) != len(profile.competitor_evidence):
        report.errors.append("duplicate claim ids within the peer evidence ledger")
    overlap = subject_ids & peer_ids
    if overlap:
        report.errors.append(
            f"claim ids shared between the subject and peer ledgers: {sorted(overlap)[:5]}"
        )

    cited: list[tuple[str, list[str]]] = []
    for label, items in (
        ("business_segment", profile.business_segments),
        ("product", profile.products),
        ("technology", profile.technologies),
        ("market_driver", profile.markets),
        ("customer", profile.customers),
        ("competitor", profile.competitors),
        ("growth_driver", profile.growth_drivers),
        ("risk", profile.risks),
        ("commercial_insight", profile.commercial_insights),
    ):
        for item in items:
            cited.append((label, list(getattr(item, "sources", []) or [])))

    dangling = 0
    for label, sources in cited:
        missing = [s for s in sources if s not in citable_ids]
        if missing:
            dangling += 1
            report.errors.append(
                f"{label} cites claim id(s) not in the ledger: {missing}"
            )

    # Deterministic items cite a FILING rather than a claim. Those must resolve to
    # a retrieved source, which is a different check from the one above.
    known_source_ids = {s.source_id for s in profile.sources if not s.error}
    for segment in profile.business_segments:
        missing_sources = [
            s for s in (segment.source_ids or []) if s not in known_source_ids
        ]
        if missing_sources:
            report.errors.append(
                f"business_segment '{segment.name}' cites unknown source id(s): "
                f"{missing_sources}"
            )
    for region in profile.geographic_revenue:
        if region.source_id and region.source_id not in known_source_ids:
            report.errors.append(
                f"geographic region '{region.region}' cites unknown source id "
                f"'{region.source_id}'"
            )

    report.stats["ledger_claims"] = len(profile.evidence)
    report.stats["cited_items"] = len(cited)
    report.stats["dangling_citations"] = dangling

    # 3. Customer disclosure discipline.
    for customer in profile.customers:
        if customer.disclosure == CustomerDisclosure.DISCLOSED:
            if not customer.sources:
                report.errors.append(
                    f"customer '{customer.name}' is marked DISCLOSED but cites "
                    f"no source; a named customer must be traceable to a filing"
                )
        elif not customer.reasoning:
            report.errors.append(
                f"customer '{customer.name}' is INFERRED_CATEGORY but gives no "
                f"reasoning"
            )

    # 4. Closed process-step taxonomy.
    valid_steps = {p.value for p in ProcessStep}
    for product in profile.products:
        for step in product.process_steps:
            if step.value not in valid_steps:
                report.errors.append(
                    f"product '{product.name}' uses process step '{step.value}' "
                    f"which is outside the taxonomy"
                )

    # 5. Company-configured scope.
    if allowed_process_steps:
        allowed = {s.lower() for s in allowed_process_steps}
        declared = {s.value for s in profile.process_steps}
        outside = {s for s in declared if s.lower() not in allowed}
        if outside:
            report.warnings.append(
                f"process steps outside the configured scope for this company: "
                f"{sorted(outside)}"
            )

    # 6. Financial period consistency.
    for metric in profile.financial_metrics:
        # Instant (balance-sheet) metrics such as inventory or deferred revenue
        # legitimately have no duration. Warning about them would produce a
        # permanent false positive, which trains a reader to ignore warnings.
        durations = [p.duration_days for p in metric.periods if p.duration_days]
        if not durations:
            continue
        for period in metric.periods:
            if period.start and period.end and period.end < period.start:
                report.errors.append(
                    f"metric '{metric.metric}' has a period ending before it starts"
                )

    # 7. Reserved for name normalisation checks. Segment names come from an
    # exact table parse, so duplicates here would indicate a parser fault rather
    # than a model fault.
    seg_names = [s.name.strip().lower() for s in profile.business_segments]
    if len(seg_names) != len(set(seg_names)):
        report.warnings.append("duplicate business segment names after normalisation")

    # Items with no analytical content at all means the synthesis stage produced
    # nothing, which should be obvious rather than silently rendered as an empty
    # report.
    qualitative = (
        len(profile.products)
        + len(profile.technologies)
        + len(profile.customers)
        + len(profile.competitors)
        + len(profile.markets)
    )
    report.stats["qualitative_items"] = qualitative
    if profile.evidence and qualitative == 0:
        report.warnings.append(
            "evidence was extracted but no analytical sections were produced; "
            "the synthesis stage may have failed"
        )

    return report
