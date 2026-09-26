"""EDGAR filing discovery and document retrieval.

CIK-driven and form-type aware from the start. Supporting ASML (a foreign private
issuer filing 20-F instead of 10-K) is therefore a configuration concern, not a
code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from ..config import FilingSettings, Settings
from ..http import CachedArtifact, HttpClient, RetrievalError

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


@dataclass
class Filing:
    """One filing in an issuer's history."""

    cik: str
    accession_number: str
    form: str
    filing_date: date | None
    report_date: date | None
    primary_document: str | None
    primary_doc_description: str | None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    # Form 8-K item numbers, e.g. '2.02,9.01'. Item 2.02 marks an earnings release.
    items: str | None = None

    @property
    def accession_nodashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def index_url(self) -> str:
        cik_int = str(int(self.cik))
        return (
            f"{ARCHIVE_BASE}/{cik_int}/{self.accession_nodashes}/"
            f"{self.accession_number}-index.htm"
        )

    @property
    def primary_document_url(self) -> str | None:
        if not self.primary_document:
            return None
        cik_int = str(int(self.cik))
        return (
            f"{ARCHIVE_BASE}/{cik_int}/{self.accession_nodashes}/"
            f"{self.primary_document}"
        )

    @property
    def is_annual(self) -> bool:
        return self.form in {"10-K", "20-F", "40-F", "10-K/A", "20-F/A"}

    @property
    def display_name(self) -> str:
        bits = [self.form]
        if self.fiscal_year:
            bits.append(f"FY{self.fiscal_year}")
        elif self.report_date:
            bits.append(str(self.report_date.year))
        return " ".join(bits)


@dataclass
class CompanySubmissions:
    """Issuer metadata plus its recent filing list."""

    cik: str
    name: str
    tickers: list[str]
    sic_description: str | None
    fiscal_year_end: str | None
    website: str | None
    investor_website: str | None
    filings: list[Filing]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _infer_fiscal_year(report_date: date | None) -> int | None:
    return report_date.year if report_date else None


def fetch_submissions(client: HttpClient, cik: str) -> CompanySubmissions:
    """Fetch issuer metadata and the *recent* filings block.

    EDGAR caps the inline ``recent`` block at ~1000 filings and paginates older
    history into separate files listed under ``filings.files``. For an
    intelligence report we need only the last few years, which always falls
    inside ``recent``; older pages are intentionally not fetched, which keeps a
    cold run to a handful of requests.
    """
    padded = str(cik).zfill(10)
    raw = client.get_json(SUBMISSIONS_URL.format(cik=padded))
    if not isinstance(raw, dict):
        raise RetrievalError(f"Unexpected submissions payload for CIK {padded}")

    recent = (raw.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filings: list[Filing] = []

    for i, form in enumerate(forms):
        report_date = _parse_date(_at(recent, "reportDate", i))
        filings.append(
            Filing(
                cik=padded,
                accession_number=_at(recent, "accessionNumber", i) or "",
                form=form,
                filing_date=_parse_date(_at(recent, "filingDate", i)),
                report_date=report_date,
                primary_document=_at(recent, "primaryDocument", i),
                primary_doc_description=_at(recent, "primaryDocDescription", i),
                fiscal_year=_infer_fiscal_year(report_date),
                items=_at(recent, "items", i),
            )
        )

    return CompanySubmissions(
        cik=padded,
        name=raw.get("name", ""),
        tickers=list(raw.get("tickers") or []),
        sic_description=raw.get("sicDescription"),
        fiscal_year_end=raw.get("fiscalYearEnd"),
        website=raw.get("website"),
        investor_website=raw.get("investorWebsite"),
        filings=filings,
    )


def _at(block: dict, key: str, i: int) -> str | None:
    """Safely index a parallel-array column of the EDGAR recent-filings block."""
    col = block.get(key)
    if isinstance(col, list) and i < len(col):
        value = col[i]
        return value if isinstance(value, str) else None
    return None


def select_filings(
    submissions: CompanySubmissions, settings: FilingSettings
) -> list[Filing]:
    """Pick the filings worth retrieving: recent annuals, quarterlies, and 8-Ks.

    Selection is by form type and recency, with the newest of each kind kept.
    This is the first and largest token-saving filter in the pipeline: it is what
    prevents us from ever considering thousands of Form 4s.
    """

    def newest_first(items: list[Filing]) -> list[Filing]:
        return sorted(items, key=lambda f: f.filing_date or date.min, reverse=True)

    annual = [f for f in submissions.filings if f.form in settings.annual_forms]
    quarterly = [f for f in submissions.filings if f.form in settings.quarterly_forms]
    events = [f for f in submissions.filings if f.form in settings.event_forms]

    selected = (
        newest_first(annual)[: settings.max_annual_filings]
        + newest_first(quarterly)[: settings.max_quarterly_filings]
        + newest_first(events)[: settings.max_event_filings]
    )
    return sorted(selected, key=lambda f: f.filing_date or date.min, reverse=True)


def fetch_filing_document(
    client: HttpClient,
    filing: Filing,
    settings: Settings,
    *,
    force_refresh: bool = False,
) -> CachedArtifact:
    """Download a filing's primary document.

    Returns the raw bytes; HTML-to-text conversion belongs to the chunking stage,
    not here, so the cache stays format-agnostic.
    """
    url = filing.primary_document_url
    if not url:
        raise RetrievalError(f"Filing {filing.accession_number} has no primary document")

    artifact = client.get(url, force_refresh=force_refresh)
    # A primary document that is suspiciously small is usually an EDGAR error
    # page rather than the filing itself; flag it instead of parsing garbage.
    if len(artifact.content) < 5_000:
        raise RetrievalError(
            f"Primary document for {filing.display_name} is only "
            f"{artifact.size_kb} KB, which is too small to be a filing"
        )
    return artifact


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
# Inline XBRL: the machine-readable fact header. It never renders to a reader, but
# once tags are stripped it becomes tens of thousands of characters of taxonomy
# URLs and element names. KLA's filing carried 253,842 characters of it, which then
# outranked real prose during chunk ranking and consumed extraction budget.
_IX_HEADER_RE = re.compile(r"<ix:header\b.*?</ix:header>", re.IGNORECASE | re.DOTALL)
_IX_HIDDEN_RE = re.compile(r"<ix:hidden\b.*?</ix:hidden>", re.IGNORECASE | re.DOTALL)
_XBRL_WRAPPER_RE = re.compile(
    r"<\?xml[^>]*\?>|<!--XBRL Document Created.*?-->", re.IGNORECASE | re.DOTALL
)
_WS_RE = re.compile(r"[ \t\xa0]+")
_BLANK_RE = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------- #
# 8-K exhibits
# --------------------------------------------------------------------------- #

# Item 2.02 of Form 8-K is "Results of Operations and Financial Condition" - the
# earnings release. This is the single most valuable non-annual source for
# commercial intelligence, and it is free and first-party on EDGAR.
_EARNINGS_ITEM = "2.02"

_INDEX_URL = ARCHIVE_BASE + "/{cik}/{accession}/index.json"

# Exhibit file names that indicate the earnings press release. EDGAR naming is
# wildly inconsistent, and a plain substring list proved too brittle: it missed Lam
# Research's 'lrcx_exhibitx991xq4x2026.htm', where 'x' is the separator instead of
# '-', '.', or nothing. This regex matches the exhibit-99 family regardless of
# separator while still requiring the 99, so an unrelated exhibit is not picked up.
_EARNINGS_EXHIBIT_RE = re.compile(
    # Exhibit 99.1 in any punctuation style: 'ex99-1', 'ex-99.1',
    # 'exhibit991...', 'lrcx_exhibitx991xq4x2026'. The lookahead permits trailing
    # digits or a filename continuation but rejects a different exhibit number
    # such as 'exhibit992'.
    r"ex(?:hibit)?[\s._\-x]*99[\s._\-x]*1(?=[\s._\-x]|\d*[a-z]|\.|$)"
    r"|ex(?:hibit)?[\s._\-x]*99(?=[\s._\-x]|\.|$)"
    r"|earnings?[\s._\-x]*(?:rele|stat|press)"
    r"|press[\s._\-x]*rele|news[\s._\-x]*rele",
    re.IGNORECASE,
)


@dataclass
class FilingExhibit:
    """A secondary document filed alongside a primary filing."""

    filing: Filing
    name: str
    url: str
    size: int

    @property
    def source_id(self) -> str:
        """Stable handle, e.g. 'sec_8k_2026-08-13_ex991'."""
        stamp = (
            self.filing.filing_date.isoformat()
            if self.filing.filing_date
            else self.filing.accession_number
        )
        return f"sec_8k_{stamp}_ex991"


def list_filing_exhibits(
    client: HttpClient, filing: Filing
) -> list[FilingExhibit]:
    """List the documents inside a filing's directory.

    EDGAR's index.json is a cheap, structured listing. This is how the earnings
    release is found: it is Exhibit 99.1 *inside* the 8-K, not the 8-K itself.
    Fetching only the 8-K's primary document - the original behaviour - retrieves
    a cover page and none of the actual commentary.
    """
    cik_int = str(int(filing.cik))
    try:
        payload = client.get_json(
            _INDEX_URL.format(cik=cik_int, accession=filing.accession_nodashes)
        )
    except RetrievalError:
        return []

    items = ((payload or {}).get("directory") or {}).get("item") or []
    exhibits: list[FilingExhibit] = []
    for item in items:
        name = str(item.get("name") or "")
        if not name.lower().endswith((".htm", ".html", ".txt")):
            continue
        if _INDEX_PAGE_RE.match(name) or name.lower().endswith("-index.html"):
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        exhibits.append(
            FilingExhibit(
                filing=filing,
                name=name,
                url=(
                    f"{ARCHIVE_BASE}/{cik_int}/{filing.accession_nodashes}/{name}"
                ),
                size=size,
            )
        )
    return exhibits


_INDEX_PAGE_RE = re.compile(r"^R\d+\.htm$", re.IGNORECASE)


def find_earnings_release(exhibits: list[FilingExhibit]) -> FilingExhibit | None:
    """Pick the earnings press release out of an 8-K's exhibits.

    Scored rather than pattern-matched, because filer naming varies widely
    ('exhibit991q32026earningsre.htm' vs 'ex99-1.htm' vs 'pressrelease.htm').
    The largest matching document wins, since the release is long and any
    accompanying image or cover exhibit is small.
    """
    scored: list[tuple[int, FilingExhibit]] = []
    for ex in exhibits:
        low = ex.name.lower()
        if _EARNINGS_EXHIBIT_RE.search(low):
            scored.append((ex.size, ex))
    if not scored:
        return None
    # Require a minimum size so a stub or image is never mistaken for the release.
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0][1]
    return best if best.size >= 10_000 else None


def is_earnings_filing(filing: Filing, items: str | None = None) -> bool:
    """True when an 8-K reports results, per its Item 2.02 designation."""
    return filing.form == "8-K" and _EARNINGS_ITEM in (items or filing.items or "")


def fetch_exhibit_document(
    client: HttpClient, exhibit: FilingExhibit, *, force_refresh: bool = False
) -> CachedArtifact:
    """Download an exhibit's content."""
    artifact = client.get(exhibit.url, force_refresh=force_refresh)
    if len(artifact.content) < 5_000:
        raise RetrievalError(
            f"Exhibit {exhibit.name} is only {artifact.size_kb} KB; too small to "
            f"be an earnings release"
        )
    return artifact


def html_to_text(html: str) -> str:
    """Convert filing HTML to readable plain text.

    Deterministic, dependency-free, and deliberately dumb: SEC filings are
    machine-generated and highly regular, so a parser adds risk without adding
    accuracy. Inline XBRL tags and financial tables are stripped because the
    numbers come from the XBRL API instead, where they are exact.

    One non-obvious step matters: Inline XBRL places a machine-readable
    ``ix:header`` at the top of the document containing every tagged fact. A naive
    tag-strip turns that into a vast blob of taxonomy text that competes with
    genuine prose for extraction budget. It is removed before anything else.
    """
    text = _IX_HEADER_RE.sub(" ", html)
    text = _IX_HIDDEN_RE.sub(" ", text)
    text = _XBRL_WRAPPER_RE.sub(" ", text)
    text = _SCRIPT_RE.sub(" ", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|h[1-6]|li)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", "\t", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)

    # Numeric character references (&#174; &#8217; &#160; &#9746; ...). These appear
    # constantly in filings and must be decoded generically: a fixed lookup table
    # missing them means the text carries a literal '&#174;' through the whole
    # pipeline, and downstream HTML rendering escapes the ampersand so a reader
    # sees "Applied Global Services&#174;" instead of the registered-trademark sign.
    def _numeric(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            code = int(raw[1:], 16) if raw.lower().startswith("x") else int(raw)
            return chr(code)
        except (ValueError, OverflowError):
            return match.group(0)

    text = re.sub(r"&#(x?[0-9A-Fa-f]+);", _numeric, text)

    # Named entities: the handful that actually appear in filings.
    for entity, char in (
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#8217;", "'"), ("&#8212;", "-"), ("&#8220;", '"'),
        ("&#8221;", '"'), ("&#160;", " "), ("&#39;", "'"),
    ):
        text = text.replace(entity, char)

    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RE.sub("\n\n", text).strip()
