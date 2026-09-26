"""Pipeline orchestration: resolve -> retrieve -> extract -> analyze -> render.

This module owns the *sequence*, not the implementation. Each stage lives in its
own module and is independently testable.

Stage 1-2 (implemented here) are fully deterministic and make zero LLM calls.
The LLM stages are deliberately absent rather than stubbed, so that nothing can
silently start spending tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from datetime import date, datetime, timezone
from pathlib import Path

from . import chunk as chunk_mod
from . import extract as extract_mod
from . import analyze as analyse_mod
from . import validate as validate_mod
from .config import Settings, load_registry, load_settings
from .chunk import Chunk
from .http import CachedArtifact, HttpClient, RetrievalError
from .llm.client import LlmClient
from . import peers as peers_mod
from .models import (
    BusinessSegment,
    CoverageSummary,
    CompanyIdentity,
    CompanyProfile,
    FinancialsExtract,
    ProcessStep,
    RevenueBreakdown,
    Source,
    SourceKind,
)
from .resolve import ResolvedCompany, resolve_company
from .sources import sec_edgar, sec_financials, sec_tables
from .validate import ValidationReport

# Map EDGAR form types onto our SourceKind enum.
_FORM_TO_KIND = {
    "10-K": SourceKind.SEC_10K,
    "10-Q": SourceKind.SEC_10Q,
    "20-F": SourceKind.SEC_20F,
    "40-F": SourceKind.SEC_40F,
    "8-K": SourceKind.SEC_8K,
    "DEF 14A": SourceKind.SEC_DEF14A,
}

# Explicit sector-role -> value-chain position. Kept as a lookup rather than
# model-generated prose, because this is a factual classification, not analysis.
_VALUE_CHAIN = {
    "equipment": "Semiconductor capital equipment (wafer fab equipment supplier)",
    "materials": "Semiconductor materials supplier",
    "foundry": "Pure-play wafer foundry",
    "idm": "Integrated device manufacturer (design + manufacturing)",
    "fabless": "Fabless semiconductor designer",
    "eda": "Electronic design automation and IP",
    "osat": "Outsourced assembly, test and packaging",
}


@dataclass
class FilingSource:
    """A retrieved artifact: citation metadata paired with its raw content.

    ``filing`` is populated for EDGAR filings and ``None`` for non-filing
    artifacts such as the XBRL company-facts payload, which is a source in its own
    right but has no accession number of its own.
    """

    source: Source
    filing: sec_edgar.Filing | None = None
    artifact: CachedArtifact | None = None

    @property
    def ok(self) -> bool:
        return self.artifact is not None and self.source.error is None


# An annual report older than this is treated as stale and flagged loudly. Set at
# 550 days: a healthy SEC registrant files annually, so an 18-month gap means at
# least one annual report is missing from the data source.
STALE_FILING_DAYS = 550


@dataclass
class RunArtifacts:
    """Everything the later stages need, produced by the deterministic stages."""

    company: ResolvedCompany
    submissions: sec_edgar.CompanySubmissions | None = None
    filings: list[FilingSource] = field(default_factory=list)
    financials: FinancialsExtract | None = None
    # Revenue breakdowns parsed from each annual report's MD&A, newest first.
    breakdowns: list[RevenueBreakdown] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    # Populated by the LLM stages; None until validation has run.
    validation: "ValidationReport | None" = None
    coverage: object | None = None
    # Set when the newest available filing is too old to be treated as current.
    newest_filing_date: date | None = None
    stale_data: bool = False

    @property
    def successful_filings(self) -> list[FilingSource]:
        return [f for f in self.filings if f.ok]

    @property
    def failed_filings(self) -> list[FilingSource]:
        return [f for f in self.filings if not f.ok]


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    return "".join(keep).strip("_")


def _source_id_for(filing: sec_edgar.Filing) -> str:
    """Stable local handle, e.g. 'sec_10k_2024' or 'sec_8k_2026-05-15'."""
    form = _slug(filing.form)
    if filing.is_annual and filing.fiscal_year:
        return f"sec_{form}_{filing.fiscal_year}"
    stamp = filing.filing_date.isoformat() if filing.filing_date else filing.accession_number
    return f"sec_{form}_{stamp}"


def stage_resolve(
    query: str, client: HttpClient, settings: Settings
) -> ResolvedCompany:
    """Stage 1: company name -> CIK."""
    return resolve_company(query, client, settings, registry=load_registry())


def stage_retrieve(
    company: ResolvedCompany,
    client: HttpClient,
    settings: Settings,
    *,
    max_filings: int | None = None,
    force_refresh: bool = False,
) -> RunArtifacts:
    """Stage 2: discover and download the filings worth keeping.

    Retrieval failures are recorded per filing and never abort the run. A company
    with three of five filings available still produces a report - it just carries
    an honest warning about what is missing.
    """
    artifacts = RunArtifacts(company=company)
    artifacts.stats = {
        "network_requests": 0,
        "cache_hits": 0,
        "filings_selected": 0,
        "filings_retrieved": 0,
        "filings_failed": 0,
    }

    submissions = sec_edgar.fetch_submissions(client, company.cik)
    artifacts.submissions = submissions

    # Recency guard.
    #
    # Measured on the newest ANNUAL REPORT, not the newest filing of any kind.
    # Infineon illustrates why: its EDGAR history contains a 2025 Form F-6 (an ADR
    # registration) but no annual report since a 2009 20-F, because it now files
    # its annual report only in Europe. Keying off the newest filing of any type
    # would have judged that company current while the substance being analysed was
    # sixteen years old.
    annual_dates = [
        f.filing_date
        for f in submissions.filings
        if f.filing_date and f.form in settings.filings.annual_forms
    ]
    newest = max(annual_dates, default=None)
    artifacts.newest_filing_date = newest
    if newest is not None:
        age_days = (date.today() - newest).days
        artifacts.stats["newest_annual_age_days"] = age_days
        if age_days > STALE_FILING_DAYS:
            years = age_days / 365
            artifacts.stale_data = True
            artifacts.warnings.append(
                f"STALE DATA: this company's most recent ANNUAL REPORT on EDGAR is "
                f"dated {newest} ({years:.1f} years old). It appears to have stopped "
                f"filing annual reports with the SEC, or files them elsewhere. Any "
                f"figure or claim below describes that period, not the company "
                f"today. Treat the report as historical."
            )
    else:
        artifacts.warnings.append(
            "No annual report (10-K / 20-F / 40-F) found on EDGAR for this company; "
            "the report rests on other filing types only."
        )

    selected = sec_edgar.select_filings(submissions, settings.filings)
    if max_filings is not None:
        # Always keep annual reports when trimming: they carry the framework's
        # substance, while 8-Ks are supplementary.
        annuals = [f for f in selected if f.is_annual]
        others = [f for f in selected if not f.is_annual]
        selected = annuals + others[: max(0, max_filings - len(annuals))]

    artifacts.stats["filings_selected"] = len(selected)

    for filing in selected:
        source_id = _source_id_for(filing)
        source = Source(
            source_id=source_id,
            kind=_FORM_TO_KIND.get(filing.form, SourceKind.SEC_8K),
            title=f"{company.name} {filing.form} filed {filing.filing_date}",
            url=filing.primary_document_url or filing.index_url,
            published=filing.filing_date,
            accession_number=filing.accession_number,
            fiscal_year=filing.fiscal_year,
        )

        try:
            artifact = sec_edgar.fetch_filing_document(
                client, filing, settings, force_refresh=force_refresh
            )
            source.content_hash = artifact.content_hash
            source.cache_path = str(
                artifact.cache_path.relative_to(settings.cache_root.parent)
            )
            source.retrieved_at = artifact.retrieved_at
            artifacts.filings.append(
                FilingSource(source=source, filing=filing, artifact=artifact)
            )
            artifacts.stats["filings_retrieved"] += 1
        except RetrievalError as exc:
            source.error = str(exc)
            artifacts.filings.append(
                FilingSource(source=source, filing=filing, artifact=None)
            )
            artifacts.stats["filings_failed"] += 1
            artifacts.warnings.append(f"{source_id}: {exc}")
            continue

        # An 8-K's primary document is a cover page. The substance - management's
        # commentary on demand, technology and customers - is Exhibit 99.1, the
        # earnings release. Without this the earnings filings cost tokens and
        # contribute nothing.
        if settings.sources.fetch_earnings_releases and sec_edgar.is_earnings_filing(
            filing
        ):
            artifacts.stats["earnings_releases_found"] = (
                artifacts.stats.get("earnings_releases_found", 0) + 1
            )
            _attach_earnings_release(
                client, filing, settings, artifacts, force_refresh=force_refresh
            )

    return artifacts


def _attach_earnings_release(
    client: HttpClient,
    filing: sec_edgar.Filing,
    settings: Settings,
    artifacts: RunArtifacts,
    *,
    force_refresh: bool = False,
) -> None:
    """Retrieve an 8-K's earnings release and register it as a citable source."""
    exhibits = sec_edgar.list_filing_exhibits(client, filing)
    release = sec_edgar.find_earnings_release(exhibits)
    if release is None:
        artifacts.warnings.append(
            f"{filing.accession_number}: earnings 8-K had no recognisable "
            f"Exhibit 99.1 press release among {len(exhibits)} document(s)"
        )
        return

    source = Source(
        source_id=release.source_id,
        kind=SourceKind.PRESS_RELEASE,
        title=f"{artifacts.company.name} earnings release {filing.filing_date}",
        url=release.url,
        published=filing.filing_date,
        accession_number=filing.accession_number,
        fiscal_year=filing.fiscal_year,
    )
    try:
        artifact = sec_edgar.fetch_exhibit_document(
            client, release, force_refresh=force_refresh
        )
    except RetrievalError as exc:
        source.error = str(exc)
        artifacts.filings.append(
            FilingSource(source=source, filing=filing, artifact=None)
        )
        artifacts.stats["filings_failed"] = artifacts.stats.get("filings_failed", 0) + 1
        artifacts.warnings.append(f"{release.source_id}: {exc}")
        return

    source.content_hash = artifact.content_hash
    source.cache_path = str(artifact.cache_path.relative_to(settings.cache_root.parent))
    source.retrieved_at = artifact.retrieved_at
    artifacts.filings.append(
        FilingSource(source=source, filing=filing, artifact=artifact)
    )
    artifacts.stats["filings_retrieved"] = artifacts.stats.get("filings_retrieved", 0) + 1
    artifacts.stats["earnings_releases_retrieved"] = (
        artifacts.stats.get("earnings_releases_retrieved", 0) + 1
    )


def stage_financials(
    artifacts: RunArtifacts, client: HttpClient, settings: Settings
) -> FinancialsExtract:
    """Stage 2b: exact financials from XBRL. Deterministic; no LLM."""
    cik = artifacts.company.cik
    submissions = artifacts.submissions
    try:
        payload = sec_financials.fetch_companyfacts(client, cik)
        extract = sec_financials.extract_financials(
            payload,
            settings,
            cik,
            # The issuer's fiscal-year-end month is what makes period labels
            # correct; without it AMAT's October year-end would be mislabelled.
            fiscal_year_end=submissions.fiscal_year_end if submissions else None,
        )
    except RetrievalError as exc:
        artifacts.warnings.append(f"XBRL financials unavailable: {exc}")
        extract = FinancialsExtract(
            cik=str(cik).zfill(10),
            entity_name=artifacts.company.name,
        )

    # Cap annual and quarterly history separately so neither silently evicts
    # the other (see sec_financials.split_periods), then refresh the descriptive
    # notes so they describe what is actually retained.
    cap = settings.financials.max_periods_per_metric
    extract.metrics = {
        name: sec_financials.refresh_notes(
            sec_financials.split_periods(metric, cap, cap)
        )
        for name, metric in extract.metrics.items()
    }
    artifacts.financials = extract

    # Register the XBRL payload as a citable source in its own right. Every
    # financial figure in the report traces back to this artifact.
    xbrl_url = sec_financials.COMPANYFACTS_URL.format(cik=str(cik).zfill(10))
    cached = client.cache.get(xbrl_url)
    artifacts.filings.append(
        FilingSource(
            source=Source(
                source_id="sec_xbrl_facts",
                kind=SourceKind.SEC_XBRL,
                title=f"{extract.entity_name} XBRL company facts",
                url=xbrl_url,
                content_hash=cached.content_hash if cached else None,
                cache_path=(
                    str(cached.cache_path.relative_to(settings.cache_root.parent))
                    if cached
                    else None
                ),
                retrieved_at=cached.retrieved_at if cached else datetime.now(timezone.utc),
            ),
            filing=None,
            artifact=cached,
        )
    )
    return extract


def build_identity(
    company: ResolvedCompany, artifacts: RunArtifacts
) -> CompanyIdentity:
    """Assemble the factual identity block.

    Every field here is either configured, taken from EDGAR metadata, or derived
    from a fixed lookup. Nothing is generated.
    """
    submissions = artifacts.submissions
    role = company.config.sector_role
    fy_end = (submissions.fiscal_year_end if submissions else None) or (
        company.config.fiscal_year_end.replace("-", "") if company.config.fiscal_year_end else None
    )
    fy_end_fmt = None
    if fy_end and len(fy_end) == 4 and fy_end.isdigit():
        fy_end_fmt = f"{fy_end[:2]}-{fy_end[2:]}"

    return CompanyIdentity(
        name=submissions.name if submissions else company.name,
        ticker=company.ticker or (submissions.tickers[0] if submissions and submissions.tickers else None),
        cik=company.cik,
        sector_role=role,
        value_chain_position=_VALUE_CHAIN.get(role or "", None),
        fiscal_year_end=fy_end_fmt,
        geographic_exposure_note=(
            "Geographic revenue split is not available from the XBRL company-facts "
            "API (segment and geographic figures are typically dimensionally tagged). "
            "It must be read from the 10-K's segment note."
        ),
    )


def stage_revenue_breakdowns(
    artifacts: RunArtifacts, settings: Settings
) -> list[RevenueBreakdown]:
    """Stage 2c: parse segment and geographic revenue tables from annual reports.

    Deterministic and free. The XBRL company-facts API omits dimensionally tagged
    facts, so segment and geographic splits are simply not available from it -
    but they sit in the MD&A in a highly regular textual form. Parsing them with
    plain Python is exact, auditable, and costs nothing, which is strictly better
    than asking an LLM to read the same table.
    """
    breakdowns: list[RevenueBreakdown] = []

    for fs in artifacts.successful_filings:
        if fs.filing is None or not fs.filing.is_annual or fs.artifact is None:
            continue
        text = sec_edgar.html_to_text(fs.artifact.text)
        breakdown = sec_tables.extract_revenue_breakdown(
            text, fs.source.source_id, fs.filing.fiscal_year
        )
        breakdowns.append(breakdown)

        # Surface the segment-table miss once, as a run-level warning, rather
        # than burying it. It is a real limitation for that filing year.
        for warning in breakdown.warnings:
            artifacts.warnings.append(f"{fs.source.source_id}: {warning}")

    # Newest fiscal year first.
    breakdowns.sort(key=lambda b: b.fiscal_year or 0, reverse=True)
    artifacts.breakdowns = breakdowns
    return breakdowns


def _merge_segments(breakdowns: list[RevenueBreakdown]) -> list[BusinessSegment]:
    """Merge segment rows across years into one list, newest figures winning.

    ``breakdowns`` must be ordered newest fiscal year first. A segment that
    appears in an older filing but not the newest (AMAT's Display business, which
    ceased to be reported separately) is retained rather than dropped, because its
    disappearance is itself signal. Its figures keep the year they came from, and
    ``revenue_note`` states that explicitly so no reader assumes it is current.
    """
    merged: dict[str, BusinessSegment] = {}
    for breakdown in breakdowns:
        year = breakdown.fiscal_year
        for seg in breakdown.by_segment:
            if seg.name not in merged:
                seg.revenue_note = (
                    f"Revenue as reported in FY{year}" if year else None
                )
                merged[seg.name] = seg

    return sorted(
        merged.values(), key=lambda s: s.revenue_millions or 0, reverse=True
    )


def build_profile(artifacts: RunArtifacts, settings: Settings) -> CompanyProfile:
    """Assemble the deterministic portion of the profile.

    Qualitative sections (products, technologies, markets, customers,
    competitors, insights) stay empty until the LLM stages run. They are omitted
    rather than filled with invented content.
    """
    identity = build_identity(artifacts.company, artifacts)
    profile = CompanyProfile(company=identity)

    for fs in artifacts.filings:
        profile.sources.append(fs.source)

    if artifacts.financials:
        profile.financial_metrics = list(artifacts.financials.metrics.values())
        profile.as_of = _latest_period_end(artifacts.financials)
        profile.warnings.extend(artifacts.financials.warnings)

    # Segment and geographic structure, from the newest annual report that
    # actually yielded a parse.
    if artifacts.breakdowns:
        newest = artifacts.breakdowns[0]
        profile.business_segments = _merge_segments(artifacts.breakdowns)
        profile.geographic_revenue = list(newest.by_geography)

    _carry_run_flags(profile, artifacts)
    profile.warnings.extend(artifacts.warnings)
    # Counted from the profile's own source list rather than the retrieval-stage
    # filing list. Peer filings are appended later by the LLM stages, so deriving
    # this from `artifacts.filings` produced the self-contradictory header
    # "Sources used: 22 of 20" - more sources used than attempted.
    profile.sources_attempted = len(profile.sources)
    profile.sources_failed = sum(1 for s in profile.sources if s.error)
    return profile


def _latest_period_end(financials: FinancialsExtract):
    ends = [
        p.end
        for m in financials.metrics.values()
        for p in m.periods
        if p.end is not None
    ]
    return max(ends) if ends else None


def _norm_name(name: str) -> str:
    """Normalise a company name for peer/subject identity comparison."""
    keep = [c.lower() if c.isalnum() else " " for c in (name or "")]
    joined = " ".join("".join(keep).split())
    # Strip EDGAR's state-of-incorporation suffix, e.g. 'applied materials inc de'.
    import re as _re

    joined = _re.sub(r"\s+(?:inc|corp|corporation|ltd|plc|nv|co)\s+[a-z]{2}$", "", joined)
    for suffix in (" inc", " corp", " corporation", " ltd", " plc", " nv", " co"):
        if joined.endswith(suffix):
            joined = joined[: -len(suffix)]
    return joined


def _merge_analysis_segments(
    deterministic: list[BusinessSegment], from_model: list[BusinessSegment]
) -> list[BusinessSegment]:
    """Enrich the deterministic segment table with the model's narrative.

    The two sources are good at different things and must not overwrite each
    other:

    * The deterministic table parse yields exact revenue, percentages and the
      fiscal year they came from.
    * The model yields what each segment actually *does*, which no table states.

    Overwriting the first with the second - the original behaviour - silently
    discarded every revenue figure and rendered the segment table as "n/a". So the
    deterministic row is the base, and only genuinely new descriptive fields are
    taken from the model.

    A segment the model names but the table did not surface is kept, with its
    revenue left null rather than invented.
    """
    def key(name: str) -> str:
        return " ".join(name.lower().replace("&", "and").split())

    def core_tokens(name: str) -> set[str]:
        """Significant words, ignoring legal noise and parenthetical acronyms."""
        cleaned = re.sub(r"\([^)]*\)", " ", key(name))
        stop = {"and", "other", "the", "segment", "group", "inc", "corp"}
        return {w for w in cleaned.split() if w not in stop and len(w) > 1}

    merged: dict[str, BusinessSegment] = {}
    for seg in deterministic:
        # Copy so the caller's list is not mutated, and so model text cannot
        # write into the deterministic original.
        merged[key(seg.name)] = seg.model_copy(deep=True)

    def find_match(name: str) -> str | None:
        """Locate the deterministic row a model label refers to.

        Exact name matching was too brittle: the model returned
        'Applied Global Services (AGS)' while the parsed table said 'Applied
        Global Services', so the report showed two rows for one segment - one with
        $6,385M and one with 'n/a'. Token overlap resolves that without inventing
        a merge: a model row only joins a parsed row when their significant words
        substantially agree.
        """
        exact = key(name)
        if exact in merged:
            return exact
        wanted = core_tokens(name)
        if not wanted:
            return None
        best_key, best_score = None, 0.0
        for candidate, seg in merged.items():
            have = core_tokens(seg.name)
            if not have:
                continue
            overlap = len(wanted & have) / len(wanted | have) if (wanted | have) else 0.0
            if overlap > best_score:
                best_key, best_score = candidate, overlap
        return best_key if best_score >= 0.55 else None

    for seg in from_model:
        matched = find_match(seg.name)
        k = matched if matched is not None else key(seg.name)
        existing = merged.get(k)
        if existing is None:
            # Segment only the model surfaced: keep it, but with no revenue.
            seg.revenue_note = seg.revenue_note or "No segment revenue row matched"
            merged[k] = seg
            continue
        if seg.description:
            existing.description = seg.description
        if seg.products:
            existing.products = seg.products
        # Union the citations so the merged row is traceable to both sources.
        for src in seg.sources:
            if src not in existing.sources:
                existing.sources.append(src)

    # Deterministic rows first, ordered by revenue; model-only rows after.
    ordered = sorted(
        merged.values(),
        key=lambda s: (s.revenue_millions is None, -(s.revenue_millions or 0)),
    )
    return ordered


def run_analysis(
    artifacts: RunArtifacts,
    profile: CompanyProfile,
    *,
    settings: Settings | None = None,
    client: LlmClient | None = None,
    progress: object | None = None,
) -> tuple[CompanyProfile, list[str]]:
    """Stages 3-5: chunk, extract, synthesise, validate.

    This is the only part of the pipeline that spends money. It is entered
    explicitly - either via ``main.py --analyze`` or this function - and never as
    a side effect of a retrieval run.

    Text selection is planned and budgeted *before* the first call, so the cost
    ceiling is a design property rather than something discovered afterwards.
    """
    settings = settings or load_settings()
    warnings: list[str] = []

    if not settings.llm.enabled:
        return profile, [
            "LLM stages are disabled (llm.enabled: false in config/settings.yaml)"
        ]

    owns_client = client is None
    client = client or LlmClient(settings)

    try:
        # --- Stage 3a: plan exactly what will be sent ----------------------- #
        all_chunks: list[Chunk] = []
        narrow_chunks: list[Chunk] = []

        # Annual report first, earnings releases second. The 10-K carries strategy
        # and risk; the releases carry management's current language on demand,
        # technology and customers, which is where product and market evidence
        # actually lives. Both are first-party and free.
        annual = [
            fs
            for fs in artifacts.successful_filings
            if fs.artifact is not None and fs.filing is not None and fs.filing.is_annual
        ]
        releases = [
            fs
            for fs in artifacts.successful_filings
            if fs.artifact is not None
            and fs.source.kind == SourceKind.PRESS_RELEASE
        ][: settings.sources.max_earnings_releases]

        for fs in annual[:1]:
            text = sec_edgar.html_to_text(fs.artifact.text)
            # Three-tier resolution: Item headings, then PART boundaries, then the
            # whole document. A Form 20-F has no Item structure, so without the
            # last tier a foreign private issuer yields no text at all.
            sections = chunk_mod.resolve_sections(text)
            if not sections:
                artifacts.warnings.append(
                    f"{fs.source.source_id}: no usable sections found; the filing "
                    f"may use unusual headings"
                )
                continue
            plan = chunk_mod.plan_extraction(
                sections,
                fs.source.source_id,
                max_chunks=settings.llm.max_extraction_chunks,
                max_total_chars=settings.llm.max_extraction_chars,
            )
            all_chunks.extend(plan.chunks)
            artifacts.stats["chunks_available"] = plan.total_chunks_available
            artifacts.stats["chunks_duplicate"] = plan.dropped_duplicates
            artifacts.stats["extraction_chars"] = plan.total_chars
            # The Business section is the densest source of product, technology
            # and competitor evidence, so it is what the coverage pass revisits.
            # Coverage pass revisits the richest sections. With the fallback tiers
            # in play the item keys differ, so anything with real prose qualifies.
            narrow_chunks.extend(
                c for c in plan.chunks if c.section_item in {"1", "7", "PART I", "PART II", "DOC"}
            )

        for fs in releases:
            text = sec_edgar.html_to_text(fs.artifact.text)
            sections = [
                chunk_mod.Section(
                    item="EX99",
                    title=f"Earnings release {fs.source.published}",
                    text=text,
                    start=0,
                    end=len(text),
                )
            ]
            plan = chunk_mod.plan_extraction(
                sections,
                fs.source.source_id,
                max_chunks=settings.llm.max_release_chunks,
                max_total_chars=settings.llm.max_release_chars,
                section_budgets={"EX99": settings.llm.max_release_chars},
            )
            all_chunks.extend(plan.chunks)
            narrow_chunks.extend(plan.chunks)
            artifacts.stats["release_chunks"] = artifacts.stats.get(
                "release_chunks", 0
            ) + len(plan.chunks)

        artifacts.stats["chunks_selected"] = len(all_chunks)

        if not all_chunks:
            warnings.append(
                "No text was selected for extraction; skipping the LLM stages."
            )
            return profile, warnings

        if callable(progress):
            progress("extract", 0, len(all_chunks), None)

        # --- Stage 3b: extract claims --------------------------------------- #
        claims, extract_warnings = extract_mod.extract_evidence(
            all_chunks,
            client=client,
            settings=settings,
            company_name=profile.company.name,
            process_steps=artifacts.company.config.relevant_process_steps,
            progress=(
                (lambda n, total, ch: progress("extract", n, total, ch))
                if callable(progress)
                else None
            ),
            narrow_chunks=narrow_chunks,
        )
        warnings.extend(f"{settings.llm.extract_model}: {w}" for w in extract_warnings)
        profile.evidence = claims
        artifacts.stats["claims_extracted"] = len(claims)

        # Measure coverage now that extraction is finished, and keep it on the
        # profile so an under-supported section is a visible, quantified limitation.
        from .coverage import measure_coverage

        coverage = measure_coverage(claims)
        artifacts.coverage = coverage
        profile.coverage = [
            CoverageSummary(
                category=c.category,
                count=c.count,
                target=c.target,
                met=c.met,
                note=None if c.met else "below target - sources may not disclose this",
            )
            for c in coverage.categories.values()
        ]
        artifacts.stats["coverage_met"] = coverage.met_count
        artifacts.stats["coverage_total"] = len(coverage.categories)

        if not claims:
            warnings.append(
                "Extraction produced no verified claims. The synthesis stage was "
                "skipped rather than asked to reason over an empty ledger."
            )
            profile.warnings.extend(warnings)
            return profile, warnings

        # --- Stage 3c: peer filings for competitive overlap ----------------- #
        # Optional and config-gated. Peer evidence is kept in its own ledger so it
        # cannot inflate the subject company's claim count, and it is the only
        # first-party route to competitor detail when the subject names none.
        # Peers are resolved for THIS subject company, not drawn from a global
        # pool. A global pool would analyse a memory maker against equipment
        # vendors, which runs cleanly and produces a wrong answer.
        run_peers = settings.sources.peers_for(profile.company.name)
        artifacts.stats["peers_configured"] = len(run_peers)
        if not run_peers and settings.sources.fetch_competitor_filings:
            warnings.append(
                "no peers configured for this company, so the competitive section "
                "rests on the subject's own disclosures only - which for many "
                "issuers name no competitor at all"
            )

        if settings.sources.fetch_competitor_filings and run_peers:
            # One shared HTTP client for all peer retrieval, so SEC pacing is
            # honoured across the whole stage rather than per request.
            # The peer pool in settings.yaml is global, so when the subject IS one
            # of the configured peers it would be listed as its own competitor -
            # which is what happened on the Lam Research run, where the report
            # named Lam Research as a competitor to itself. Filter the subject out.
            # Guard against a peer_map entry that accidentally lists the subject
            # as its own peer, which happened with a global pool and produced a
            # report naming the company as its own competitor.
            subject_keys = {
                _norm_name(profile.company.name),
                _norm_name(artifacts.company.name),
                (profile.company.ticker or "").upper(),
                (artifacts.company.ticker or "").upper(),
            }
            subject_keys.discard("")
            peers_for_run = [
                p
                for p in run_peers
                if _norm_name(p.name) not in subject_keys
                and (p.ticker or "").upper() not in subject_keys
            ]
            if len(peers_for_run) != len(run_peers):
                artifacts.warnings.append(
                    "peer configuration listed the subject company as its own peer; "
                    "excluded"
                )

            with HttpClient(settings) as peer_client:
                peer_docs, peer_warnings = peers_mod.fetch_peer_documents(
                    peers_for_run,
                    client=peer_client,
                    settings=settings,
                )
                peer_chunks = peers_mod.plan_peer_chunks(peer_docs, settings=settings)
            warnings.extend(peer_warnings)
            artifacts.stats["peer_documents"] = len(peer_docs)
            artifacts.stats["peer_chunks"] = len(peer_chunks)
            if peer_chunks:
                if callable(progress):
                    progress("peers", 0, len(peer_chunks), None)
                peer_claims, peer_claim_warnings = extract_mod.extract_peer_evidence(
                    peer_chunks,
                    client=client,
                    settings=settings,
                    subject_name=profile.company.name,
                    progress=(
                        (lambda n, total, ch: progress("peers", n, total, ch))
                        if callable(progress)
                        else None
                    ),
                )
                warnings.extend(peer_claim_warnings)
                profile.competitor_evidence = peer_claims
                artifacts.stats["peer_claims"] = len(peer_claims)
                # Register peer filings as citable sources so provenance is visible.
                for doc in peer_docs:
                    if doc.ok:
                        profile.sources.append(doc.source)

        # --- Stage 4: synthesise -------------------------------------------- #
        if callable(progress):
            progress("analyse", 0, 1, None)

        analysis, analyse_warnings = analyse_mod.analyse(
            claims,
            client=client,
            settings=settings,
            company_name=profile.company.name,
            process_steps=artifacts.company.config.relevant_process_steps,
            financials=profile.financial_metrics,
            segments=profile.business_segments,
            geography=profile.geographic_revenue,
            breakdowns=artifacts.breakdowns,
            peer_evidence=profile.competitor_evidence,
        )
        warnings.extend(analyse_warnings)

        # Enrich rather than replace: the model adds narrative, the table parse
        # keeps the numbers. See _merge_analysis_segments.
        profile.business_segments = _merge_analysis_segments(
            profile.business_segments, analysis.get("business_segments", [])
        )
        profile.products = analysis.get("products", [])
        profile.technologies = analysis.get("technologies", [])
        profile.markets = analysis.get("markets", [])
        profile.customers = analysis.get("customers", [])
        profile.competitors = analysis.get("competitors", [])
        # Set `basis` from the configuration, not from the model. Whether a peer
        # relationship was disclosed by the subject or chosen by a reviewer is a
        # fact about how this report was built, so it must not depend on the model
        # reproducing a schema hint correctly.
        basis_by_key = {}
        for peer in run_peers:
            basis_by_key[_norm_name(peer.name)] = peer.basis
            if peer.ticker:
                basis_by_key[peer.ticker.upper()] = peer.basis
        for comp in profile.competitors:
            comp.basis = basis_by_key.get(
                _norm_name(comp.name), basis_by_key.get(comp.name.upper(), "industry")
            )
        profile.growth_drivers = analysis.get("growth_drivers", [])
        profile.risks = analysis.get("risks", [])
        profile.commercial_insights = analysis.get("commercial_insights", [])

        if analysis.get("value_chain_position"):
            # Only override the deterministic classification when the model's
            # version is supported by a citation.
            profile.company.value_chain_position = analysis["value_chain_position"]

        # Process steps evidenced by the analysis, plus any named directly in the
        # claim ledger. Products are not the only place they appear, so the ledger
        # scan stops a supported process step from being invisible merely because
        # the synthesis stage did not attach it to a product.
        steps: list[ProcessStep] = []
        for product in profile.products:
            for step in product.process_steps:
                if step not in steps:
                    steps.append(step)
        for step in analyse_mod.process_steps_from_claims(
            claims, artifacts.company.config.relevant_process_steps
        ):
            if step not in steps:
                steps.append(step)
        profile.process_steps = steps

        # --- Stage 5: validate ---------------------------------------------- #
        report = validate_mod.validate_profile(
            profile,
            allowed_process_steps=artifacts.company.config.relevant_process_steps,
        )
        artifacts.validation = report
        warnings.extend(report.errors)
        warnings.extend(report.warnings)
        # Coverage shortfalls are reported, not hidden. An unmet target means the
        # sources did not support the category, which the reader should know.
        warnings.extend(coverage.notes())
        artifacts.stats["validation_errors"] = len(report.errors)

        if callable(progress):
            progress("analyse", 1, 1, None)

        profile.warnings.extend(warnings)
        # Final accounting. Peer filings are appended during the LLM stages, so
        # both counters must be recomputed from the completed source list here -
        # otherwise the header reads "Sources used: 22 of 20", claiming more sources
        # were used than were ever attempted.
        _refresh_source_counts(profile)
        return profile, warnings
    finally:
        if owns_client:
            client.close()


def _refresh_source_counts(profile: CompanyProfile) -> None:
    """Recompute source tallies from the profile's own source list."""
    profile.sources_attempted = len(profile.sources)
    profile.sources_failed = sum(1 for s in profile.sources if s.error)


def _carry_run_flags(profile: CompanyProfile, artifacts: RunArtifacts) -> None:
    """Copy run-level diagnostics onto the profile.

    Called at the end of both the deterministic and the analysis paths, because a
    retrieval-only run must still record staleness. Setting these only inside the
    LLM branch left every profile reporting no annual-report date at all, which
    silently disabled the stale-data banner the guard exists to raise.
    """
    profile.stale_data = artifacts.stale_data
    profile.newest_filing_date = artifacts.newest_filing_date


def run(
    query: str,
    *,
    settings: Settings | None = None,
    max_filings: int | None = None,
    force_refresh: bool = False,
) -> tuple[CompanyProfile, RunArtifacts]:
    """Run the deterministic pipeline end to end.

    Returns the profile plus the raw run artifacts (filing text, XBRL payload) so
    that later stages can work from cache without re-retrieving anything.

    This function makes no LLM calls. Pass the result to :func:`run_analysis` to
    add the analytical sections.
    """
    settings = settings or load_settings()

    with HttpClient(settings) as client:
        company = stage_resolve(query, client, settings)
        artifacts = stage_retrieve(
            company,
            client,
            settings,
            max_filings=max_filings,
            force_refresh=force_refresh,
        )
        stage_financials(artifacts, client, settings)
        stage_revenue_breakdowns(artifacts, settings)

        artifacts.stats["network_requests"] = client.network_requests
        artifacts.stats["cache_hits"] = client.cache_hits

        profile = build_profile(artifacts, settings)
        _carry_run_flags(profile, artifacts)

    return profile, artifacts


def write_profile(profile: CompanyProfile, path: Path) -> Path:
    """Persist the structured profile as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        profile.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
    )
    return path


def default_profile_path(profile: CompanyProfile) -> Path:
    """data/profiles/<ticker-or-slug>.json - stable across runs, so incremental
    updates can diff against the previous extraction."""
    from .config import project_root

    key = (profile.company.ticker or profile.company.name).lower()
    return project_root() / "data" / "profiles" / f"{_slug(key)}.json"
