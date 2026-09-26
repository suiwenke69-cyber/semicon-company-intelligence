"""Core schemas for the Semiconductor Company Intelligence Agent.

Design rule that governs this whole file:

    Numbers come from XBRL. Prose comes from the LLM. The two never mix.

Consequently there are two distinct families of model here:

* ``XbrlValue`` / ``FinancialMetric`` - exact, deterministic, machine-extracted.
  Every figure is traceable to a taxonomy tag and an SEC accession number.
* ``Evidence`` / ``*Claim`` - qualitative, LLM-extracted, and therefore required
  to carry a verbatim excerpt plus a ``source_id``. If it has no source, it is
  not a fact and cannot be modelled as one.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enumerations - closed sets. The extraction model SELECTS from these; it is
# never permitted to invent a new process step.
# --------------------------------------------------------------------------- #


class ClaimType(str, Enum):
    """The epistemic status of a statement. Rendered distinctly in the report."""

    FACT = "FACT"                  # directly stated in a cited first-party source
    ANALYSIS = "ANALYSIS"          # reasoned from two or more FACT claims
    INFERENCE = "INFERENCE"        # reasoned with an explicit, stated chain


class SectorRole(str, Enum):
    EQUIPMENT = "equipment"
    MATERIALS = "materials"
    FOUNDRY = "foundry"
    IDM = "idm"
    FABLESS = "fabless"
    EDA = "eda"
    OSAT = "osat"


class ProcessStep(str, Enum):
    """Closed taxonomy. Mirrors the specification's equipment process-step list."""

    DEPOSITION = "Deposition"
    ETCH = "Etch"
    LITHOGRAPHY = "Lithography"
    PROCESS_CONTROL = "Process Control"
    INSPECTION = "Inspection"
    METROLOGY = "Metrology"
    IMPLANTATION = "Implantation"
    CMP = "CMP"
    ADVANCED_PACKAGING = "Advanced Packaging"
    THERMAL_PROCESSING = "Thermal Processing"
    MATERIALS_ENGINEERING = "Materials Engineering"
    CLEANING = "Cleaning"
    WAFER_PREP = "Wafer Preparation"
    TEST = "Test"


class CustomerDisclosure(str, Enum):
    """Distinguishes a named customer in a filing from a category we reasoned about.

    This distinction is the specification's explicit anti-hallucination
    requirement for customer relationships.
    """

    DISCLOSED = "DISCLOSED"                  # named in a cited source; citation required
    INFERRED_CATEGORY = "INFERRED_CATEGORY"  # e.g. "leading-edge foundries"; reasoning required


class SourceKind(str, Enum):
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    SEC_20F = "20-F"
    SEC_40F = "40-F"
    SEC_8K = "8-K"
    SEC_DEF14A = "DEF 14A"
    SEC_XBRL = "SEC-XBRL"
    PRESS_RELEASE = "Press Release"
    INVESTOR_PRESENTATION = "Investor Presentation"
    PRODUCT_PAGE = "Product Page"
    INDUSTRY = "Industry Source"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------- #
# Source tracking
# --------------------------------------------------------------------------- #


class Source(BaseModel):
    """A retrieved artifact. One per filing, XBRL payload, or web page."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Stable local handle, e.g. 'sec_10k_2024'.")
    kind: SourceKind
    title: str
    url: str
    retrieved_at: datetime = Field(default_factory=_utcnow)
    published: date | None = None
    accession_number: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    # sha256 of the stored artifact, so a cached document can be proven unchanged.
    content_hash: str | None = None
    # Relative path inside cache/, for provenance and debugging.
    cache_path: str | None = None
    # Populated when a source was attempted but failed; the report surfaces these
    # rather than silently degrading.
    error: str | None = None


# --------------------------------------------------------------------------- #
# Deterministic financials (XBRL) - no LLM involvement, ever
# --------------------------------------------------------------------------- #


class XbrlValue(BaseModel):
    """A single as-reported XBRL fact, preserved without interpretation."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    taxonomy: str = Field(description="'us-gaap' or the company's own extension namespace.")
    tag: str
    label: str | None = None
    unit: str
    value: float | None = None
    start: date | None = None
    end: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    accession_number: str | None = None
    filed: date | None = None
    frame: str | None = Field(
        default=None,
        description="SEC calendar frame. The presence of a frame means this is the "
        "authoritative value SEC selected for that period.",
    )
    duration_days: int | None = Field(
        default=None,
        description="Length of the reporting period. Distinguishes a quarter "
        "(~91 days) from a fiscal year (~365).",
    )
    period_label: str | None = Field(
        default=None,
        description="Human-readable period, e.g. 'FY2024' or 'Q3 FY2024'.",
    )

    @property
    def is_duration(self) -> bool:
        """True for flow metrics (revenue over a period) vs. stock (balance sheet)."""
        return self.start is not None and self.end is not None


class FinancialMetric(BaseModel):
    """One metric, deduplicated across filings, newest period first."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    unit: str
    concept: str = Field(description="Taxonomy tag actually used, e.g. 'us-gaap:Revenues'.")
    periods: list[XbrlValue] = Field(default_factory=list)
    yoy_growth_pct: float | None = None
    note: str | None = None

    @property
    def latest(self) -> XbrlValue | None:
        """Most recent period of any length."""
        return self.periods[0] if self.periods else None

    @property
    def latest_annual(self) -> XbrlValue | None:
        """Most recent full-year period.

        Headline figures should lead with this rather than whichever quarter
        happens to be freshest, so that a revenue number is never silently
        compared against a different metric's annual figure.
        """
        for p in self.periods:
            if p.duration_days and p.duration_days > 300:
                return p
        return None

    @property
    def latest_quarter(self) -> XbrlValue | None:
        for p in self.periods:
            if p.duration_days and p.duration_days <= 300:
                return p
        return None

    @property
    def preferred_period(self) -> XbrlValue | None:
        """Annual if the metric has any, else the most recent period."""
        return self.latest_annual or self.latest


class FinancialsExtract(BaseModel):
    """Output of the deterministic XBRL step. Written to financials.json."""

    model_config = ConfigDict(extra="forbid")

    cik: str
    entity_name: str
    fiscal_year_end: str | None = None
    extracted_at: datetime = Field(default_factory=_utcnow)
    form_types_seen: list[str] = Field(default_factory=list)
    metrics: dict[str, FinancialMetric] = Field(default_factory=dict)
    # Concepts discovered in the company's own extension namespace that look
    # financial. Useful for later segment/geographic work.
    extension_concepts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Evidence layer - the contract between extraction and analysis
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    """A single extracted claim with mandatory provenance.

    ``analysis.py`` receives a list of these and NOTHING else. Its own outputs must
    cite ``claim_id`` values that exist in this list, which makes citation
    validation a mechanical assertion rather than a hope.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str = Field(description="Normalised claim, a single assertion.")
    claim_type: ClaimType = ClaimType.FACT
    source_id: str
    locator: str | None = Field(
        default=None, description="Section reference, e.g. 'Item 1 Business, para 12'."
    )
    excerpt: str | None = Field(
        default=None,
        description="Short verbatim supporting excerpt, attributed. Capped by "
        "settings.report.excerpt_max_chars; never bulk filing text.",
    )
    confidence: Confidence = Confidence.HIGH
    topic: str | None = Field(default=None, description="Framework section this feeds.")
    # Populated for non-FACT claims: which existing claims this was reasoned from.
    derived_from: list[str] = Field(default_factory=list)


# Topics the extraction stage may assign. A closed set, so a model cannot invent
# a topic that the synthesis stage then silently ignores.
EXTRACTION_TOPICS: tuple[str, ...] = (
    "company_overview",
    "product",
    "technology",
    "process_step",
    "customer",
    "competitor",
    "market_driver",
    "financial_commentary",
    "risk",
    "strategy",
    "geography",
)


class ExtractedClaim(BaseModel):
    """Raw claim as returned by the extraction model, before normalisation.

    Deliberately minimal. Anything the model is not required to produce is a
    token we do not pay for and a field it cannot get wrong.
    """

    model_config = ConfigDict(extra="ignore")

    claim: str
    quote: str | None = None
    topic: str | None = None
    confidence: Confidence = Confidence.MEDIUM


class ExtractionResult(BaseModel):
    """Normalised output of extracting one chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_id: str
    locator: str
    claims: list[Evidence] = Field(default_factory=list)
    dropped_invalid: int = 0
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Framework sections
# --------------------------------------------------------------------------- #


class ProductItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: str | None = None
    description: str | None = None
    process_steps: list[ProcessStep] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="claim_id references.")


class TechnologyItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    differentiator: str | None = None
    roadmap_note: str | None = None
    sources: list[str] = Field(default_factory=list)


class BusinessSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    products: list[str] = Field(default_factory=list)
    revenue_note: str | None = None
    # Deterministically parsed from the filing's own MD&A segment table.
    revenue_millions: float | None = None
    revenue_pct_of_total: float | None = None
    revenue_prior_year_millions: float | None = None
    revenue_change_pct: float | None = None
    # Two different kinds of provenance, deliberately kept apart:
    #   sources    - claim ids from the evidence ledger (LLM-derived content)
    #   source_ids - filing source ids (deterministic table parses)
    # Conflating them made the validator report a table parse citing
    # 'sec_10_k_2025' as a dangling claim id.
    sources: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class GeographicRevenue(BaseModel):
    """Revenue by customer-facility region, parsed from the 10-K's geographic table.

    Deterministic: this is a table extraction, not an estimate. Region labels are
    the issuer's own, so AMAT's "Asia Pacific" subtotal is preserved as reported
    rather than recomputed.
    """

    model_config = ConfigDict(extra="forbid")

    region: str
    revenue_millions: float | None = None
    pct_of_total: float | None = None
    prior_year_millions: float | None = None
    change_pct: float | None = None
    is_subtotal: bool = False
    source_id: str | None = None
    locator: str | None = None


class RevenueBreakdown(BaseModel):
    """The outcome of deterministic table extraction for one filing."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    fiscal_year: int | None = None
    total_revenue_millions: float | None = None
    by_segment: list[BusinessSegment] = Field(default_factory=list)
    by_geography: list[GeographicRevenue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MarketDriver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    driver: str
    relevance: str
    company_exposure: str | None = None
    claim_type: ClaimType = ClaimType.ANALYSIS
    sources: list[str] = Field(default_factory=list)


class CustomerInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    disclosure: CustomerDisclosure
    category: str | None = None
    requirements: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    value_created: str | None = None
    reasoning: str | None = Field(
        default=None, description="Required for INFERRED_CATEGORY entries."
    )
    sources: list[str] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def _disclosed_requires_source(cls, v: list[str], info) -> list[str]:
        # Note: cross-field validators on optional disclosure are enforced in
        # validate.py where the whole profile is available; this is a cheap guard.
        return v


class Competitor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # How this peer relationship is established, which a reader needs in order to
    # weigh it:
    #   disclosed - the subject company's own filing names this peer
    #   industry  - the reviewer chose it from the industry grouping a human
    #               selected; the overlap is evidenced by the peer's own filing but
    #               the pairing itself is a judgement call, not a disclosure
    basis: str = "industry"
    product_overlap: str | None = None
    technology_positioning: str | None = None
    market_exposure: str | None = None
    strengths: list[str] = Field(default_factory=list)
    potential_weaknesses: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: str
    category: str | None = None
    claim_type: ClaimType = ClaimType.FACT
    mitigation_note: str | None = None
    sources: list[str] = Field(default_factory=list)


class CommercialInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    insight: str
    kind: Literal[
        "growth_opportunity",
        "technology_opportunity",
        "customer_opportunity",
        "competitive_threat",
        "business_risk",
    ]
    claim_type: ClaimType = ClaimType.ANALYSIS
    reasoning: str | None = None
    sources: list[str] = Field(default_factory=list)


class CompanyIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ticker: str | None = None
    cik: str | None = None
    sector_role: SectorRole | None = None
    value_chain_position: str | None = None
    headquarters: str | None = None
    fiscal_year_end: str | None = None
    geographic_exposure_note: str | None = None


# --------------------------------------------------------------------------- #
# Top-level profile
# --------------------------------------------------------------------------- #


class CoverageSummary(BaseModel):
    """Measured evidence coverage per analytical category.

    Exposed in the profile so that an under-supported section is a visible,
    quantified limitation rather than something a reader has to infer from an
    empty list.
    """

    model_config = ConfigDict(extra="forbid")

    category: str
    count: int
    target: int
    met: bool
    note: str | None = None


class CompanyProfile(BaseModel):
    """The structured output. Mirrors the specification's JSON contract exactly."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1.0"
    generated_at: datetime = Field(default_factory=_utcnow)
    as_of: date | None = Field(
        default=None, description="Latest period covered by the underlying data."
    )
    company: CompanyIdentity
    business_segments: list[BusinessSegment] = Field(default_factory=list)
    geographic_revenue: list[GeographicRevenue] = Field(
        default_factory=list,
        description="Revenue by customer-facility region, from the newest annual "
        "report's MD&A table. Deterministically parsed, not estimated.",
    )
    products: list[ProductItem] = Field(default_factory=list)
    technologies: list[TechnologyItem] = Field(default_factory=list)
    process_steps: list[ProcessStep] = Field(default_factory=list)
    markets: list[MarketDriver] = Field(default_factory=list)
    customers: list[CustomerInsight] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    financial_metrics: list[FinancialMetric] = Field(default_factory=list)
    growth_drivers: list[MarketDriver] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    commercial_insights: list[CommercialInsight] = Field(default_factory=list)
    coverage: list[CoverageSummary] = Field(
        default_factory=list,
        description="Per-category evidence coverage against advisory targets.",
    )
    competitor_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Competitive-overlap evidence drawn from PEER filings. Kept "
        "separate from the main ledger so it never inflates the subject company's "
        "own claim count, and typed INFERENCE because a peer describing itself is "
        "not a factual claim about the subject.",
    )
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Full claim ledger. Kept in the profile so any downstream "
        "assertion can be traced to a source without re-reading documents.",
    )
    # Diagnostics: what was attempted but unavailable. Never silently dropped.
    sources_attempted: int = 0
    sources_failed: int = 0
    # Set when the newest available filing is too old to represent the company as
    # it is today. Some US-registered foreign issuers effectively stop filing.
    stale_data: bool = False
    newest_filing_date: date | None = None
    warnings: list[str] = Field(default_factory=list)
