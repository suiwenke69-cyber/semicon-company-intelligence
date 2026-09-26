"""Static HTML front end.

Deliberately not a web application. The profiles change only when analysis is
re-run, so generating a self-contained HTML page per company gives a browsable
interface with:

* **no server** - the output opens by double-clicking,
* **no new dependency** - Jinja2 is already used for the Markdown report,
* **no build step** - `--web` runs in the same process as the analysis.

The tradeoff, stated plainly: a static site cannot trigger a new analysis. That is
the correct trade for a portfolio artefact, because the interesting thing to show a
reviewer is the *output* and its traceability, not a button that spends money.

Every page is standalone, so any single file can be sent to someone as-is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import project_root
from .formatting import format_money, format_percent
from .models import CompanyProfile

DEFAULT_SITE_DIR = "site"


@dataclass
class CompanySummary:
    """The subset of a profile the index page needs, so the index stays small."""

    ticker: str | None
    name: str
    slug: str
    href: str
    value_chain: str | None
    revenue_label: str | None
    revenue_growth: float | None
    rd_growth: float | None
    claims: int
    sources_used: int
    sources_attempted: int
    coverage_met: int
    coverage_total: int
    competitors: list[str] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)
    generated_at: datetime | None = None


def _metric(profile: CompanyProfile, name: str):
    for m in profile.financial_metrics:
        if m.metric == name:
            return m
    return None


def slug_for(profile: CompanyProfile) -> str:
    key = profile.company.ticker or profile.company.name
    return "".join(c.lower() if c.isalnum() else "-" for c in key).strip("-")


def build_summary(profile: CompanyProfile) -> CompanySummary:
    """Reduce a profile to the fields the dashboard displays."""
    rev = _metric(profile, "revenue")
    rd = _metric(profile, "rd_expense")
    latest = rev.latest_annual if rev else None

    return CompanySummary(
        ticker=profile.company.ticker,
        name=profile.company.name,
        slug=slug_for(profile),
        href=f"{slug_for(profile)}.html",
        value_chain=profile.company.value_chain_position,
        revenue_label=(
            format_money(latest.value, rev.unit) if latest and rev else None
        ),
        revenue_growth=rev.yoy_growth_pct if rev else None,
        rd_growth=rd.yoy_growth_pct if rd else None,
        claims=len(profile.evidence),
        sources_used=sum(1 for s in profile.sources if not s.error),
        sources_attempted=profile.sources_attempted or len(profile.sources),
        coverage_met=sum(1 for c in profile.coverage if c.met),
        coverage_total=len(profile.coverage),
        competitors=[c.name for c in profile.competitors],
        segments=[
            (s.name, format_money(s.revenue_millions, "millions"))
            for s in profile.business_segments
            if s.revenue_millions
        ],
        generated_at=profile.generated_at,
    )


def load_profiles(profile_dir: Path) -> list[CompanyProfile]:
    """Load every valid profile in a directory, skipping anything unreadable.

    A corrupt or half-written profile must not take down the whole site, so parse
    failures are reported to the caller's console rather than raised.
    """
    profiles: list[CompanyProfile] = []
    for path in sorted(profile_dir.glob("*.json")):
        try:
            profiles.append(
                CompanyProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
            )
        except Exception as exc:  # noqa: BLE001 - a bad file should not be fatal
            print(f"  skipped {path.name}: {type(exc).__name__}: {exc}")
    return profiles


def build_environment(template_dir: Path | None = None) -> Environment:
    """Jinja environment with the same formatting rules as the Markdown report.

    Sharing the filters matters: if the HTML said "$28,368M" and the Markdown said
    "28.4B" for the same figure, a reader would reasonably distrust both.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_dir or project_root() / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,  # HTML output, unlike the Markdown report
    )
    env.filters["money"] = format_money
    env.filters["pct"] = format_percent
    env.filters["signed_pct"] = lambda v: format_percent(v, signed=True)
    env.filters["millions"] = (
        lambda v: "n/a" if v is None else f"{v:,.0f}M"
    )
    return env


def generate_site(
    profile_dir: Path | None = None,
    out_dir: Path | None = None,
    template_dir: Path | None = None,
) -> tuple[Path, list[CompanyProfile]]:
    """Render every profile plus an index into a static site directory."""
    root = project_root()
    profile_dir = profile_dir or (root / "data" / "profiles")
    out_dir = out_dir or (root / DEFAULT_SITE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(profile_dir)
    if not profiles:
        raise FileNotFoundError(
            f"No readable company profiles in {profile_dir}. Run an analysis first:\n"
            f'  python main.py "Applied Materials" --analyze'
        )

    env = build_environment(template_dir)
    generated = datetime.now(timezone.utc)

    # The stylesheet is inlined into every page so each HTML file is standalone and
    # can be sent to someone as a single attachment.
    css_path = (template_dir or (root / "templates")) / "shared.css"
    shared_css = css_path.read_text(encoding="utf-8")
    context_common = {"shared_css": shared_css, "generated": generated}

    company_tpl = env.get_template("company.html.j2")
    for profile in profiles:
        summary = build_summary(profile)
        html = company_tpl.render(
            profile=profile,
            summary=summary,
            markdown_href=None,
            **context_common,
        )
        (out_dir / summary.href).write_text(html, encoding="utf-8")

    summaries = sorted(
        (build_summary(p) for p in profiles), key=lambda s: s.name.lower()
    )
    index_tpl = env.get_template("index.html.j2")
    (out_dir / "index.html").write_text(
        index_tpl.render(
            companies=summaries,
            total_claims=sum(s.claims for s in summaries),
            total_sources=sum(s.sources_used for s in summaries),
            **context_common,
        ),
        encoding="utf-8",
    )

    return out_dir, profiles
