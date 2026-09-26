"""Deterministic financial extraction from SEC's XBRL API.

    No LLM touches a number in this project.

This module is the reason that guarantee holds. It reads
``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` - a free,
structured, machine-readable dump of every fact the issuer has tagged - and
reduces it to a small set of metrics with exact values and full provenance.

What this buys us:

* **Zero hallucination risk on figures.** Values are integers from the filing.
* **Zero token cost.** Roughly 60% of the context a "paste the 10-K" approach
  would need simply never enters a prompt.
* **Restatement awareness.** Every candidate fact records its accession number
  and filing date, so a restatement is visibly resolved rather than silently lost.

Scope limitation, stated plainly: segment and geographic revenue breakdowns are
often tagged only with dimensions, and ``companyfacts`` exposes just the
undimensioned facts. Those figures must come from the filings' own tables, which
is a later stage's job. We do not fabricate them here.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from datetime import date, datetime
from typing import Any

from ..config import FinancialSettings, Settings
from ..http import HttpClient, RetrievalError
from ..models import FinancialMetric, FinancialsExtract, XbrlValue

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Unit preference: simple monetary units beat per-share ratios and pure numbers.
_UNIT_PRIORITY = ("USD", "usd", "shares", "USD/shares")

# Form priority when the same period is reported more than once.
# Annual reports are authoritative for full-year figures; 10-Q for quarterlies.
_FORM_PRIORITY = {"10-K": 3, "20-F": 3, "40-F": 3, "10-Q": 2, "6-K": 1, "8-K": 0}

# Fiscal period ordering within a fiscal year, for building period labels.
_PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}

# Rough day counts used to classify a duration fact.
_QUARTER_DAYS = (80, 100)
_ANNUAL_DAYS = (340, 380)


def fetch_companyfacts(client: HttpClient, cik: str) -> dict[str, Any]:
    """Retrieve the raw companyfacts payload.

    This file is large (AMAT is ~4.5 MB uncompressed, larger issuers reach tens of
    MB), which is precisely why it is cached to disk on first fetch and never
    re-downloaded. One request per company, reused across every metric.
    """
    padded = str(cik).zfill(10)
    raw = client.get_json(COMPANYFACTS_URL.format(cik=padded))
    if not isinstance(raw, dict) or "facts" not in raw:
        raise RetrievalError(f"No XBRL facts available for CIK {padded}")
    return raw


# --------------------------------------------------------------------------- #
# Fact extraction helpers
# --------------------------------------------------------------------------- #


def _span_days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (d1 - d0).days


def _period_label(
    start: str | None,
    end: str | None,
    fye_month: int | None,
    span: int | None,
) -> tuple[str | None, int | None, int | None, str | None]:
    """Build a period label from the DATES, never from the filing's ``fy`` field.

    This matters more than it looks. XBRL's ``fy``/``fp`` elements describe the
    *filing*, not the period being reported: the prior-year comparative inside a
    FY2024 10-Q still carries ``fy=2024`` while ending in October 2023. Trusting
    it produced two different periods both labelled "FY2024", and it is the same
    trap that corrupts fiscal-quarter labelling generally.

    So we derive the fiscal period from the period end date and the issuer's
    fiscal-year-end month:

        AMAT, FYE month = 10 (October)
          2025-01-26 -> Q1 FY2025      (fiscal year began Nov 2024)
          2025-10-26 -> FY2025         (363-day span)

    Returns ``(label, fiscal_year, quarter, length_class)`` where length_class is
    ``'annual'``, ``'quarter'`` or ``None``.
    """
    if not end:
        return None, None, None, None
    try:
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return None, None, None, None

    if span is None:
        # Instant fact (balance sheet): label by fiscal year only.
        fy = _fiscal_year_of(end_date, fye_month)
        return f"FY{fy} (instant)", fy, None, None

    length_class = (
        "annual"
        if _ANNUAL_DAYS[0] <= span <= _ANNUAL_DAYS[1]
        else "quarter"
        if _QUARTER_DAYS[0] <= span <= _QUARTER_DAYS[1]
        else None
    )

    if length_class == "annual":
        fy = _fiscal_year_of(end_date, fye_month)
        return f"FY{fy}", fy, None, "annual"

    if length_class == "quarter":
        fy, quarter = _fiscal_quarter_of(end_date, fye_month)
        return f"Q{quarter} FY{fy}", fy, quarter, "quarter"

    fy = _fiscal_year_of(end_date, fye_month)
    return f"{span}d to FY{fy}", fy, None, None


def _fiscal_year_of(end_date: date, fye_month: int | None) -> int:
    """Fiscal year that a period ending on ``end_date`` belongs to.

    With an October fiscal-year end, a period ending 2025-10-26 is FY2025, while
    one ending 2025-11-15 already belongs to FY2026.
    """
    if not fye_month:
        return end_date.year
    return end_date.year if end_date.month <= fye_month else end_date.year + 1


def _fiscal_quarter_of(end_date: date, fye_month: int | None) -> tuple[int, int]:
    """Fiscal (year, quarter) pair for a period end date."""
    if not fye_month:
        return end_date.year, (end_date.month - 1) // 3 + 1

    fy = _fiscal_year_of(end_date, fye_month)
    # Months elapsed since the fiscal year began (the month after the FYE month).
    start_month = fye_month % 12 + 1
    months_in = (end_date.month - start_month) % 12 + 1
    quarter = min(4, (months_in - 1) // 3 + 1)
    return fy, quarter


def _select_units(tag_block: dict[str, Any]) -> tuple[str, list[dict]] | None:
    """Choose the best unit bucket for a tag, preferring plain USD."""
    units = tag_block.get("units") or {}
    for preferred in _UNIT_PRIORITY:
        if preferred in units:
            return preferred, units[preferred]
    if units:
        key = next(iter(units))
        return key, units[key]
    return None


def _dedupe_facts(
    raw_facts: list[dict],
    metric: str,
    taxonomy: str,
    tag: str,
    unit: str,
    label: str | None,
    fye_month: int | None,
    *,
    prefer_duration: bool,
) -> list[XbrlValue]:
    """Reduce raw facts to one authoritative value per reporting period.

    Deduplication rule, applied in order:

    1. Classify each fact as duration (flow) or instant (stock) and keep only the
       kind this metric wants. This is what stops a quarterly revenue figure from
       being mixed with a full-year one.
    2. Group by period. When SEC has assigned a ``frame``, that frame is the
       canonical period identity; otherwise key on (start, end).
    3. Within a period, prefer annual-report provenance, then a frame-bearing
       fact, then the most recently filed value. That last tiebreak is what makes
       restatements resolve correctly instead of arbitrarily.
    """
    candidates: list[tuple[tuple, XbrlValue]] = []

    for f in raw_facts:
        start, end = f.get("start"), f.get("end")
        span = _span_days(start, end)
        is_duration = start is not None and end is not None

        if prefer_duration and not is_duration:
            continue
        if not prefer_duration and is_duration:
            continue

        # For flows, discard cumulative year-to-date spans (e.g. 9-month) so that
        # only genuine quarters and full years survive.
        if prefer_duration:
            if span is None:
                continue
            is_q = _QUARTER_DAYS[0] <= span <= _QUARTER_DAYS[1]
            is_y = _ANNUAL_DAYS[0] <= span <= _ANNUAL_DAYS[1]
            if not (is_q or is_y):
                continue

        value = f.get("val")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        form = f.get("form")
        frame = f.get("frame")
        filed = f.get("filed")

        # When SEC assigned a frame, that frame IS the canonical period identity.
        # Otherwise fall back to the raw span.
        period_key: tuple = ("frame", frame) if frame else ("span", start, end)
        rank = (
            _FORM_PRIORITY.get(str(form), 0),
            1 if frame else 0,
            filed or "",
        )

        period_label, fiscal_year, quarter, _length = _period_label(
            start, end, fye_month, span
        )

        candidates.append(
            (
                period_key,
                rank,
                XbrlValue(
                    metric=metric,
                    taxonomy=taxonomy,
                    tag=tag,
                    label=label,
                    unit=unit,
                    value=numeric,
                    start=_to_date(start),
                    end=_to_date(end),
                    # Deliberately derived from the period end date, NOT taken
                    # from the filing's ``fy`` field (see _period_label).
                    fiscal_year=fiscal_year,
                    fiscal_period=f"Q{quarter}" if quarter else None,
                    form=form,
                    accession_number=f.get("accn"),
                    filed=_to_date(filed),
                    frame=frame,
                    duration_days=span,
                    period_label=period_label,
                ),
            )
        )

    # One winner per canonical period: highest rank wins, where rank is
    # (form priority, has-frame, filing date) compared lexicographically.
    best: dict[tuple, tuple[tuple, XbrlValue]] = {}
    for period_key, rank, value in candidates:
        current = best.get(period_key)
        if current is None or rank > current[0]:
            best[period_key] = (rank, value)

    # A span can legitimately carry two different frames (SEC occasionally issues
    # both a quarterly and a cumulative frame for the same dates). Collapse those
    # onto the span, preferring annual-report provenance.
    by_span: dict[tuple, XbrlValue] = {}
    for _, value in best.values():
        span_key = (value.start, value.end)
        existing = by_span.get(span_key)
        if existing is None or _FORM_PRIORITY.get(
            str(value.form), 0
        ) > _FORM_PRIORITY.get(str(existing.form), 0):
            by_span[span_key] = value

    deduped = list(by_span.values())
    deduped.sort(key=lambda v: (v.end or date.min), reverse=True)
    return deduped


def _to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _compute_yoy(periods: list[XbrlValue], lookback: int) -> float | None:
    """Year-over-year growth, comparing strictly like-for-like periods.

    The invariant enforced here: **a quarter is only ever compared with a
    quarter, and a year with a year.** Violating it produces confidently wrong
    numbers - during development this compared a 90-day operating cash flow of
    $1,686M against a 363-day figure of $7,958M and reported "-8.3% YoY", which
    is worse than reporting nothing.

    Annual periods are used when the issuer has at least two, because that is the
    comparison a reader expects and it avoids fiscal-quarter edge cases. Otherwise
    the most recent matching-duration quarters are used.
    """
    annuals = [p for p in periods if p.duration_days and p.duration_days > 300]
    quarters = [p for p in periods if p.duration_days and p.duration_days <= 300]
    pool = annuals if len(annuals) >= 2 else quarters
    if len(pool) < 2:
        return None

    current = pool[0]
    if current.end is None or current.start is None or not current.value:
        return None
    current_len = (current.end - current.start).days

    best: XbrlValue | None = None
    best_delta = 10_000
    for cand in pool[1:]:
        if cand.end is None or cand.start is None or not cand.value:
            continue
        cand_len = (cand.end - cand.start).days
        # Duration must match within a few days: this is the guard that prevents
        # the quarter-versus-year error.
        if abs(cand_len - current_len) > 10:
            continue
        delta = abs((current.end - cand.end).days - 365)
        if delta < best_delta:
            best_delta, best = delta, cand

    # Reject a "prior year" that is not actually about a year away.
    if best is None or best_delta > 45 or not best.value:
        return None
    return round((current.value - best.value) / abs(best.value) * 100, 1)


def _year_over_year_pair(
    periods: list[XbrlValue],
) -> tuple[XbrlValue, XbrlValue] | None:
    """Return the (current, prior-year) pair behind ``yoy_growth_pct``.

    Exposed so the report can show its work rather than asserting a percentage
    with no visible basis.
    """
    annuals = [p for p in periods if p.duration_days and p.duration_days > 300]
    quarters = [p for p in periods if p.duration_days and p.duration_days <= 300]
    pool = annuals if len(annuals) >= 2 else quarters
    if len(pool) < 2:
        return None
    current = pool[0]
    if current.end is None or current.start is None:
        return None
    current_len = (current.end - current.start).days

    best: XbrlValue | None = None
    best_delta = 10_000
    for cand in pool[1:]:
        if cand.end is None or cand.start is None:
            continue
        if abs((cand.end - cand.start).days - current_len) > 10:
            continue
        delta = abs((current.end - cand.end).days - 365)
        if delta < best_delta:
            best_delta, best = delta, cand
    if best is None or best_delta > 45:
        return None
    return current, best


# --------------------------------------------------------------------------- #
# Metric assembly
# --------------------------------------------------------------------------- #

# Regex used to surface the issuer's own extension concepts that look financial.
_FINANCIAL_HINT = re.compile(
    r"(revenue|sales|income|expense|margin|backlog|bookings|cash|inventory|"
    r"segment|deferred|capex|capital)",
    re.IGNORECASE,
)


def parse_fiscal_year_end(raw: str | None) -> int | None:
    """Parse EDGAR's ``fiscalYearEnd`` (e.g. '1026') into a month number.

    Returns 10 for AMAT (October), 8 for Micron (August), 12 for a calendar-year
    filer. ``None`` means we could not tell, and period labels degrade to calendar
    years rather than guessing.
    """
    if not raw or len(raw) != 4 or not raw.isdigit():
        return None
    month = int(raw[:2])
    return month if 1 <= month <= 12 else None


def extract_financials(
    facts_payload: dict[str, Any],
    settings: Settings,
    cik: str,
    *,
    fiscal_year_end: str | None = None,
) -> FinancialsExtract:
    """Reduce a companyfacts payload to the metrics the framework needs.

    All periods are retained (subject to a generous safety cap); the annual and
    quarterly split is applied later by :func:`split_periods`, so that capping
    never silently drops recent quarters.
    """
    fin_cfg: FinancialSettings = settings.financials
    facts_by_taxonomy: dict[str, dict] = facts_payload.get("facts") or {}
    us_gaap = facts_by_taxonomy.get("us-gaap", {})
    fye_month = parse_fiscal_year_end(fiscal_year_end)

    result = FinancialsExtract(
        cik=str(cik).zfill(10),
        entity_name=facts_payload.get("entityName", ""),
        fiscal_year_end=fiscal_year_end,
        form_types_seen=[],
    )
    if fye_month is None:
        result.warnings.append(
            "Fiscal-year-end month could not be determined; period labels fall "
            "back to calendar years, which may differ from the issuer's fiscal "
            "year (e.g. AMAT's FY ends in October)."
        )

    forms_seen: set[str] = set()

    for metric, preferred_tags in fin_cfg.concept_preferences.items():
        chosen_block = None
        chosen_tag = None

        # Try preferred us-gaap concepts first, in priority order.
        for tag in preferred_tags:
            block = us_gaap.get(tag)
            if block and block.get("units"):
                chosen_block, chosen_tag = block, tag
                break

        # Fallback: sweep the issuer's taxonomy for a plausible match. This is
        # what keeps a company whose tagging differs from our defaults working
        # without a code change.
        if chosen_block is None:
            keyword = metric.split("_")[0]
            for tag, block in us_gaap.items():
                if keyword.lower() in tag.lower() and block.get("units"):
                    chosen_block, chosen_tag = block, tag
                    result.warnings.append(
                        f"'{metric}': no preferred concept matched; fell back to "
                        f"us-gaap:{tag}"
                    )
                    break

        if chosen_block is None or chosen_tag is None:
            continue

        picked = _select_units(chosen_block)
        if picked is None:
            continue
        unit, raw_facts = picked

        prefer_duration = metric not in {"inventory", "deferred_revenue"}
        # EPS and share counts are legitimately reported as such.
        if metric == "diluted_eps":
            prefer_duration = True

        periods = _dedupe_facts(
            raw_facts,
            metric=metric,
            taxonomy="us-gaap",
            tag=chosen_tag,
            unit=unit,
            label=chosen_block.get("label"),
            fye_month=fye_month,
            prefer_duration=prefer_duration,
        )
        if not periods:
            result.warnings.append(
                f"'{metric}': concept us-gaap:{chosen_tag} present but no usable "
                f"quarterly/annual periods"
            )
            continue

        for p in periods:
            if p.form:
                forms_seen.add(p.form)

        # A generous safety cap only; the report-facing split into annual and
        # quarterly history happens in split_periods().
        periods = periods[: fin_cfg.max_periods_per_metric * 3]

        result.metrics[metric] = FinancialMetric(
            metric=metric,
            unit=unit,
            concept=f"us-gaap:{chosen_tag}",
            periods=periods,
            yoy_growth_pct=_compute_yoy(periods, fin_cfg.growth_lookback_periods),
        )

    # Record extension concepts worth investigating for segment/geographic work.
    for taxonomy, tags in facts_by_taxonomy.items():
        if taxonomy in {"us-gaap", "dei", "srt"}:
            continue
        for tag in tags:
            if _FINANCIAL_HINT.search(tag):
                result.extension_concepts.append(f"{taxonomy}:{tag}")

    result.form_types_seen = sorted(forms_seen)

    if not result.metrics:
        result.warnings.append(
            "No financial metrics could be extracted from XBRL. The issuer may "
            "tag figures unconventionally, or may not file XBRL at all."
        )

    # A compact diagnostic: how many periods we hold per metric, and whether the
    # figures look annual or quarterly. Useful when reviewing extraction quality.
    for name, metric in result.metrics.items():
        if metric.periods:
            annual, quarterly = count_period_kinds(metric)
            metric.note = f"{annual} annual, {quarterly} quarterly"

    return result


def count_period_kinds(metric: FinancialMetric) -> tuple[int, int]:
    """Split a metric's periods into (annual, quarterly) counts."""
    annual = sum(
        1 for p in metric.periods if p.duration_days and p.duration_days > 300
    )
    return annual, len(metric.periods) - annual


def split_periods(
    metric: FinancialMetric, max_annual: int, max_quarterly: int
) -> FinancialMetric:
    """Return a copy of ``metric`` with annual and quarterly history capped separately.

    Why this exists: a single flat cap applied to annual and quarterly periods
    mixed together will silently evict recent quarters once a metric accumulates
    enough annual history. That is exactly what happened during development -
    operating cash flow stopped at Q1 FY2026 while revenue reached Q3 FY2026,
    purely as an artifact of truncation, not of the data.

    Fractional `duration_days` is the discriminator; no reliance on the filing's
    fiscal-period fields.
    """
    annual = [p for p in metric.periods if p.duration_days and p.duration_days > 300]
    quarterly = [
        p for p in metric.periods if not (p.duration_days and p.duration_days > 300)
    ]
    kept = sorted(
        annual[:max_annual] + quarterly[:max_quarterly],
        key=lambda p: (p.end or date.min),
        reverse=True,
    )
    return metric.model_copy(update={"periods": kept})


def latest_annual(metric: FinancialMetric) -> XbrlValue | None:
    """Most recent full-year period, which is what a financial overview should lead with."""
    for p in metric.periods:
        if p.duration_days and p.duration_days > 300:
            return p
    return None


def latest_quarter(metric: FinancialMetric) -> XbrlValue | None:
    """Most recent quarterly period."""
    for p in metric.periods:
        if p.duration_days and p.duration_days <= 300:
            return p
    return None


def refresh_notes(metric: FinancialMetric) -> FinancialMetric:
    """Recompute the descriptive ``note`` from the periods actually retained.

    Must be called after any capping. The note was previously computed before
    truncation, so it advertised "18 annual, 18 quarterly" for a metric storing
    12 and 12 - a small lie that would undermine trust in the whole report.
    """
    annual, quarterly = count_period_kinds(metric)
    bits = []
    if annual:
        bits.append(f"{annual} annual")
    if quarterly:
        bits.append(f"{quarterly} quarterly")
    return metric.model_copy(update={"note": ", ".join(bits) or "no periods"})


def summarise_growth(metric: FinancialMetric) -> str | None:
    """Human-readable growth line for the report, e.g. 'FY2024: 33.0B (+2.3% YoY)'."""
    latest = metric.latest
    if latest is None or latest.value is None:
        return None
    scale, suffix = _scale(latest.value)
    line = f"{latest.period_label or 'latest'}: {latest.value / scale:,.1f}{suffix}"
    if metric.yoy_growth_pct is not None:
        line += f" ({metric.yoy_growth_pct:+.1f}% YoY)"
    return line


def _scale(value: float) -> tuple[float, str]:
    """Scale a raw dollar figure to a readable unit."""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return 1e9, "B"
    if magnitude >= 1e6:
        return 1e6, "M"
    if magnitude >= 1e3:
        return 1e3, "K"
    return 1.0, ""


def median_period_span(metric: FinancialMetric) -> float | None:
    spans = [p.duration_days for p in metric.periods if p.duration_days]
    return statistics.median(spans) if spans else None
