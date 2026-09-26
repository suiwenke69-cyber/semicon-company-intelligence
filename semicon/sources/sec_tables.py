"""Deterministic extraction of revenue breakdown tables from filing text.

The XBRL company-facts API exposes only *undimensioned* facts, so segment and
geographic revenue never appear there - they are tagged with dimensions that the
API omits. Those tables do, however, sit in the 10-K's MD&A in a highly regular
textual form. Parsing them with plain Python is:

* exact (the figures come straight from the filing),
* free (no tokens),
* auditable (each value keeps its source_id and locator).

That combination is strictly better than asking an LLM to read the table, so no
LLM is involved here at all.

Expected input shape, which is what our HTML-to-text layer produces::

    Net revenue by segment for the periods presented were as follows:

    Change
    2025 2024 2025 over 2024

    (In millions, except percentages)
    Semiconductor Systems $ 20,798 73% $ 19,911 73% 4 %
    Applied Global Services 6,385 23% 6,225 23% 3 %
    Corporate and Other 1,185 4% 1,040 4% 14 %
    Total $ 28,368 100% $ 27,176 100% 4 %
"""

from __future__ import annotations

import re

from ..models import (
    BusinessSegment,
    GeographicRevenue,
    RevenueBreakdown,
)

# A currency amount: "28,368", "1,185", "8,529". Negative values are parenthesised.
_MONEY_RE = re.compile(r"^\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?$|^\(?\d+\.\d+\)?$")

# Percentages, including the odd negative form "(16) %".
_PCT_RE = re.compile(r"^\(?-?\d+(?:\.\d+)?\)?\s*%$")

# Lead-in sentences for revenue tables, as an ORDERED list of patterns.
#
# Order matters and is the whole point. A single loose pattern matched AMAT's
# sentence "revenue has not been recognized ... Backlog by reportable segment"
# and parsed the wrong table entirely, silently shifting every segment figure. So
# the most specific, historically reliable phrasing is tried first, and looser
# patterns are fallbacks only.
_SEGMENT_LEADINS: tuple[re.Pattern[str], ...] = (
    # AMAT: "Net revenue by segment for the periods presented were as follows:"
    re.compile(
        r"Net (?:revenue|sales) by segment for the periods presented "
        r"(?:was|were) as follows:",
        re.I | re.DOTALL,
    ),
    # "Revenue by reportable segment:" / "Net sales by segment:"
    re.compile(
        r"(?:Net\s+)?(?:revenue|sales)\s+by\s+(?:reportable\s+)?segment\s*:",
        re.I,
    ),
    # "The following table presents revenue disaggregated by segment:"
    re.compile(
        r"revenue\s+disaggregated\s+by\s+(?:reportable\s+)?segment[^:]{0,40}:",
        re.I | re.DOTALL,
    ),
)

# Not parsed, deliberately.
#
# Some equipment vendors report no segment revenue at all - they operate as a
# single reportable segment - and instead disaggregate revenue between systems and
# services. Lam Research does exactly this.
#
# An attempt to parse that table was made and reverted. Lam presents it in
# THOUSANDS while its geographic table is in MILLIONS, uses column headers that
# split across lines ("Year Ended" / "June 28," / "2026 June 29," / "2025"), and
# the table has no labelled total row, so the collector ran on into the following
# gross-margin table. The result was a segment figure of $11,725,308M - a number
# wrong by six orders of magnitude, presented with the same confidence as a correct
# one.
#
# Silently emitting a wrong revenue figure is far worse than reporting the table as
# absent, so this stays unimplemented until unit detection and table-boundary
# handling can be done properly. The shortfall is surfaced as a warning instead.
_DISAGGREGATION_LEADIN = re.compile(
    r"The following table presents (?:the Company's )?(?:our )?(?:total )?revenue\s+"
    r"disaggregated\s+between\s+(?:system|systems)\s+and\s+"
    r"(?:customer[-\s]support|service|support)[^:]{0,60}:",
    re.I | re.DOTALL,
)

# Geographic revenue. Issuers decorate this sentence heavily. AMAT writes "Net
# sales by geographic region, determined by the location of customers' facilities
# to which products were shipped, were as follows:" while Lam Research writes "The
# following table presents our total revenue and revenue disaggregated by
# geographic region:". Both end in 'geographic region' followed by a colon or
# 'as follows'.
_GEO_LEADIN = re.compile(
    r"(?:revenue|sales)[^:]{0,200}?(?:by|disaggregated\s+by)\s+"
    r"geographic\s+(?:region|area|location)[^:]{0,120}?"
    r"(?::|(?:was|were)\s+as\s+follows)",
    re.I | re.DOTALL,
)

# Rows that terminate a table.
_TOTAL_LABEL = re.compile(r"^total\b", re.I)

# Four-digit year, used to spot column-header rows masquerading as data rows.
_YEAR_TOKEN = re.compile(r"^(19|20)\d{2}$")

# Boilerplate that must never be mistaken for a table row.
_NOISE = re.compile(
    r"^(table of contents|change|\d{4}\s+\d{4}|in millions.*|"
    r"\(in millions.*|\d+\s*$|the change.*)$",
    re.I,
)

# Geographic region labels we accept, so a stray prose line is not parsed as a
# region. Issuers name these slightly differently, hence a permissive set.
_KNOWN_REGIONS = {
    "china", "korea", "south korea", "taiwan", "japan", "singapore",
    "southeast asia", "asia pacific", "asia-pacific", "asia", "united states",
    "u.s.", "us", "north america", "europe", "emea", "americas", "others",
    "rest of world", "rest of asia", "united states of america",
}

# Subtotal rows that are the sum of the rows above them. Any sum check must
# exclude these, or the constituent regions get counted twice.
_SUBTOTAL_REGIONS = {"asia pacific", "asia-pacific", "asia", "americas", "emea"}

# Trailing punctuation and placeholder glyphs that filings attach to row labels.
# AMAT's FY2023 China row renders as "China -" because the change column carries
# a dash meaning "no change"; without stripping it the region fails to match.
_LABEL_JUNK = re.compile(r"[\s\-–—.,:;*()]+$")


# Table units. Filings report in millions or thousands, and assuming millions
# produced figures wrong by a factor of 1000: KLA's geographic revenue rendered as
# "North America 1,757,337M" when the company's total revenue is about $13.6
# billion. A confidently wrong number is worse than a missing one, so the unit is
# read from the table's own header and every parsed amount is scaled to millions.
# Filer conventions vary: "(In thousands)", "(in millions)", "(In thousands,
# except percentages)", and even "Revenue (in millions)". The opening paren is
# optional and no trailing punctuation is required.
_UNIT_RE = re.compile(r"\(?\s*in\s+(thousands|millions|billions)\b", re.IGNORECASE)
_UNIT_SCALE = {"thousands": 0.001, "millions": 1.0, "billions": 1000.0}


def _detect_unit_scale(text: str, start: int, window: int = 900) -> float:
    """Scale factor converting this table's amounts into millions.

    Defaults to millions, which is the most common convention, but honours an
    explicit "(In thousands)" or "(In billions)" header.
    """
    window_text = text[start : start + window]
    match = _UNIT_RE.search(window_text)
    if not match:
        return 1.0
    return _UNIT_SCALE.get(match.group(1).lower(), 1.0)


def _to_float(token: str) -> float | None:
    """Parse a money token, honouring the parenthesised-negative convention."""
    t = token.strip()
    if not t:
        return None
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(",", "").replace("%", "").strip()
    try:
        value = float(t)
    except ValueError:
        return None
    return -value if negative else value


def _parse_row(line: str) -> tuple[str, list[float], list[float]] | None:
    """Split one table row into (label, money values, percentages).

    Returns ``None`` for headings, prose, and anything that does not look like a
    data row. Being strict here is what stops prose sentences from being parsed as
    financial rows.

    Two traps in real filing text drive the logic below, both discovered by
    parsing AMAT's actual 10-K rather than a tidy fixture:

    1. Negative changes render as ``(16) %`` with a space, so a stray ``%``
       follows a bare number. Naive tokenising reads the ``16`` as money.
    2. Some rows omit the thousands separator: ``Europe 962 3% 1,443 5% (33) %``.
       Worse, the loose numbers ``962 3`` are adjacent, so splitting on
       whitespace fuses them into a bogus ``9623``. The figure is only
       recoverable by realising that ``962`` is the value immediately preceding
       the ``3%`` that belongs to it.

    So we anchor on the percentage markers: each ``%`` claims the nearest number
    to its left that is not already claimed. Any remaining numbers are amounts
    whose percentage was absent (commonly the prior-year column). All numbers are
    kept in document order afterwards.
    """
    s = line.strip().replace("\\$", "$")
    if not s or _NOISE.match(s):
        return None

    # Tokenise with a custom splitter rather than str.split. Two adjustments:
    #   * detach '%' so it becomes its own token,
    #   * split *loose* digit runs that fused together ("962 3" -> "962","3").
    # Comma-grouped amounts ("28,368") and decimals ("8.66") must stay intact, so
    # any token containing ',' or '.' is left exactly as it is.
    raw_tokens = s.replace("%", " % ").split()
    tokens: list[str] = []
    for tok in raw_tokens:
        if tok == "%" or "," in tok or "." in tok:
            tokens.append(tok)
            continue
        parts = re.findall(r"\(?\d+\)?", tok)
        tokens.extend(parts or [tok])

    numbers: list[dict] = []
    pct_indices: list[int] = []
    label_words: list[str] = []

    for tok in tokens:
        cleaned = tok.replace("$", "").strip()
        if cleaned == "%":
            pct_indices.append(len(numbers))
            continue
        if not cleaned:
            continue
        numeric = _MONEY_RE.match(cleaned) or re.match(r"^\(?\d+\)?$", cleaned)
        if numeric:
            value = _to_float(cleaned)
            if value is not None:
                numbers.append({"value": value, "is_pct": False})
            continue
        label_words.append(cleaned)

    if not numbers:
        return None

    # Each '%' claims the nearest number to its left that is still unclaimed.
    claimed: set[int] = set()
    for idx in pct_indices:
        for j in range(idx - 1, -1, -1):
            if j not in claimed:
                numbers[j]["is_pct"] = True
                claimed.add(j)
                break

    amounts = [n["value"] for n in numbers if not n["is_pct"]]
    pcts = [n["value"] for n in numbers if n["is_pct"]]
    if not amounts:
        return None

    label = " ".join(label_words).strip()
    if not label or len(label) < 2:
        return None
    # Reject labels that are clearly prose fragments rather than row labels.
    if label.endswith((".", ":")) and len(label.split()) > 6:
        return None

    # Structural validation. A column header such as
    # "2025 2024 2025 over 2024" otherwise parses as a row labelled "over" with
    # four amounts, which then corrupts the parts-sum-to-total check and causes
    # the genuinely correct rows to be discarded.
    #
    # The header is identified by what it is: every "amount" is a four-digit year
    # and no percentage is present. Deliberately NOT identified by label length
    # or by amount magnitude - an earlier version used both heuristics and
    # wrongly discarded real rows ("China" is only 5 characters; AMAT's Display
    # segment was only $885M).
    if not pcts and amounts and all(
        1900 <= abs(a) <= 2100 for a in amounts
    ):
        return None
    if not pcts:
        return None

    return label, amounts, pcts


def _first_leadin_match(
    text: str, leadin: re.Pattern[str] | tuple[re.Pattern[str], ...] | list[re.Pattern[str]]
):
    """Find the first lead-in that matches, honouring pattern order.

    Ordered rather than alternated: a specific phrasing must win over a loose one,
    otherwise a loose pattern can match an unrelated sentence earlier in the
    document and the wrong table gets parsed.
    """
    patterns = leadin if isinstance(leadin, (tuple, list)) else (leadin,)
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _iter_table_rows(
    text: str,
    leadin: re.Pattern[str] | tuple[re.Pattern[str], ...] | list[re.Pattern[str]],
    *,
    max_rows: int = 30,
) -> list[str]:
    """Collect the lines of the first table following a lead-in sentence.

    Stops at the total row (inclusive) or after a blank-prose runaway, so a
    missing table cannot cause the whole document to be scanned.
    """
    match = _first_leadin_match(text, leadin)
    if not match:
        return []

    tail = text[match.end():]
    rows: list[str] = []
    started = False

    for raw_line in tail.split("\n")[: max_rows * 3]:
        line = raw_line.strip()
        if not line:
            continue
        if _parse_row(line) is None:
            # Header/noise lines before the table starts are expected; once we
            # are inside the table, prose means the table has ended.
            if started and not _NOISE.match(line) and len(line.split()) > 8:
                break
            continue
        started = True
        rows.append(line)
        if _TOTAL_LABEL.match(line):
            break
        if len(rows) >= max_rows:
            break

    return rows


def parse_segment_table(
    text: str,
    source_id: str,
    fiscal_year: int | None = None,
    warnings: list[str] | None = None,
) -> list[BusinessSegment]:
    """Parse the MD&A 'Net revenue by segment' table."""
    rows = _iter_table_rows(text, _SEGMENT_LEADINS)
    segments: list[BusinessSegment] = []
    total: float | None = None

    leadin = _first_leadin_match(text, _SEGMENT_LEADINS)
    scale = _detect_unit_scale(text, leadin.start()) if leadin else 1.0

    for line in rows:
        parsed = _parse_row(line)
        if parsed is None:
            continue
        label, amounts, pcts = parsed
        if _TOTAL_LABEL.match(label):
            total = amounts[0] * scale if amounts else None
            continue
        segments.append(
            BusinessSegment(
                name=label,
                revenue_millions=amounts[0] * scale if amounts else None,
                revenue_pct_of_total=pcts[0] if pcts else None,
                revenue_prior_year_millions=(
                    amounts[1] * scale if len(amounts) > 1 else None
                ),
                # Document-level provenance, not a claim citation.
                source_ids=[source_id],
            )
        )

    if not segments:
        return []

    # Sanity check: the parts should sum to the reported total. A mismatch means
    # we probably parsed something that is not really the segment table - but we
    # report it rather than silently discarding rows that parsed cleanly, because
    # a missing "Corporate and Other" row legitimately breaks the sum.
    if total is not None:
        parts = sum(s.revenue_millions or 0 for s in segments)
        if abs(parts - total) > max(2.0, total * 0.01):
            if warnings is not None:
                warnings.append(
                    f"Revenue rows sum to {parts:,.0f}M but the reported total is "
                    f"{total:,.0f}M - a reconciling row may be missing."
                )


    for s in segments:
        if s.revenue_millions and s.revenue_prior_year_millions:
            prior = s.revenue_prior_year_millions
            if prior:
                s.revenue_change_pct = round(
                    (s.revenue_millions - prior) / abs(prior) * 100, 1
                )
    return segments


def parse_geographic_table(
    text: str, source_id: str, fiscal_year: int | None = None
) -> tuple[list[GeographicRevenue], float | None]:
    """Parse the MD&A 'Net revenue by geographic region' table.

    Returns the region rows plus the reported total. Subtotal rows (e.g. AMAT's
    "Asia Pacific") are preserved and flagged, never silently summed into.
    """
    rows = _iter_table_rows(text, _GEO_LEADIN)
    regions: list[GeographicRevenue] = []
    total: float | None = None

    leadin = _first_leadin_match(text, _GEO_LEADIN)
    scale = _detect_unit_scale(text, leadin.start()) if leadin else 1.0

    for line in rows:
        parsed = _parse_row(line)
        if parsed is None:
            continue
        label, amounts, pcts = parsed
        if _TOTAL_LABEL.match(label):
            total = amounts[0] * scale if amounts else None
            continue
        # Strip trailing punctuation/placeholder glyphs before matching a region.
        label = _LABEL_JUNK.sub("", label).strip()
        if label.lower() not in _KNOWN_REGIONS:
            # Not a region we recognise: skip rather than emit an invention.
            continue
        regions.append(
            GeographicRevenue(
                region=label,
                # Only the first two columns are the current and prior fiscal
                # year. Later "amounts" are change percentages that the source
                # table mixes into the same numeric stream, so they are not
                # interpreted as revenue.
                revenue_millions=amounts[0] * scale if amounts else None,
                pct_of_total=pcts[0] if pcts else None,
                prior_year_millions=(
                    amounts[1] * scale if len(amounts) > 1 else None
                ),
                is_subtotal=label.lower() in _SUBTOTAL_REGIONS,
                source_id=source_id,
                locator="MD&A: Net revenue by geographic region",
            )
        )

    # Derive change % where we have both years.
    for r in regions:
        if r.revenue_millions is not None and r.prior_year_millions:
            r.change_pct = round(
                (r.revenue_millions - r.prior_year_millions)
                / abs(r.prior_year_millions)
                * 100,
                1,
            )

    return regions, total


def extract_revenue_breakdown(
    text: str, source_id: str, fiscal_year: int | None = None
) -> RevenueBreakdown:
    """Extract both breakdowns from one filing's text."""
    result = RevenueBreakdown(source_id=source_id, fiscal_year=fiscal_year)

    segments = parse_segment_table(
        text, source_id, fiscal_year, warnings=result.warnings
    )
    regions, geo_total = parse_geographic_table(text, source_id, fiscal_year)

    result.by_segment = segments
    result.by_geography = regions

    # Cross-check that the region rows reconcile to the reported total. Subtotals
    # MUST be excluded: AMAT's "Asia Pacific" is the sum of five regions above it,
    # so including it double-counts and makes a correct parse look broken.
    constituent = [r for r in regions if not r.is_subtotal]
    if constituent:
        parts = sum(r.revenue_millions or 0 for r in constituent)
        if geo_total is not None and abs(parts - geo_total) > max(2.0, geo_total * 0.01):
            result.warnings.append(
                f"Geographic regions sum to {parts:,.0f}M but the reported total "
                f"is {geo_total:,.0f}M; a region row may have been missed."
            )

    result.total_revenue_millions = geo_total or (
        sum(s.revenue_millions or 0 for s in segments) if segments else None
    )

    if not segments:
        result.warnings.append(
            "No segment revenue table found. The issuer may report a single "
            "reportable segment, or present revenue differently."
        )
    if not regions:
        result.warnings.append(
            "No geographic revenue table found in this filing."
        )

    return result
