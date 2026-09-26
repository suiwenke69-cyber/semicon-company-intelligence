"""Configuration loading for the agent.

Two files, both plain YAML, both committed:

* ``config/settings.yaml``  - runtime knobs, budgets, HTTP policy.
* ``config/companies.yaml`` - the single extension point for adding a company.

Nothing here reads the network or the environment except the optional API key,
which is read lazily so that steps 1-2 never require one.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


def project_root() -> Path:
    """Repo root, resolved from this file's location (semicon/config.py -> ../)."""
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# settings.yaml
# --------------------------------------------------------------------------- #


class HttpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = 30
    max_retries: int = 3
    min_seconds_between_requests: float = 0.15


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "cache"
    max_age_days: int = 30


class FilingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annual_forms: list[str] = Field(default_factory=lambda: ["10-K", "20-F", "40-F"])
    quarterly_forms: list[str] = Field(default_factory=lambda: ["10-Q", "6-K"])
    event_forms: list[str] = Field(default_factory=lambda: ["8-K"])
    max_annual_filings: int = 3
    max_quarterly_filings: int = 4
    max_event_filings: int = 8


class FinancialSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_preferences: dict[str, list[str]] = Field(default_factory=dict)
    max_periods_per_metric: int = 12
    growth_lookback_periods: int = 4


class LlmSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    extract_model: str = "gpt-4o-mini"
    synth_model: str = "gpt-4o"
    max_output_tokens: int = 4000
    temperature: float = 0
    run_token_budget: int = 60_000
    # Realistic completion size used for budget projection. max_output_tokens caps
    # any single call; this is what the guard actually budgets with, so the guard
    # estimates true spend rather than aborting affordable runs half way through.
    budget_output_estimate: int = 1200
    # Ceilings on how much filing text may ever be sent to the model. The planner
    # stops at whichever binds first, so a pathological filing cannot produce an
    # unbounded prompt.
    max_extraction_chunks: int = 18
    max_extraction_chars: int = 40_000
    # Earnings releases are compact and dense with commercial language, so they
    # get their own small budget rather than competing with the 10-K for space.
    max_release_chunks: int = 6
    max_release_chars: int = 14_000
    price_per_mtok_input: dict[str, float] = Field(default_factory=dict)
    price_per_mtok_output: dict[str, float] = Field(default_factory=dict)


class ReportSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    excerpt_max_chars: int = 300


class CompetitorSourceConfig(BaseModel):
    """A peer company whose own filings are used for competitive overlap.

    Kept as configuration because it is a judgement call, not a derivation: the
    pipeline cannot know which companies a given issuer competes with, and
    guessing would be exactly the unsupported claim this project exists to avoid.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    cik: str | None = None
    ticker: str | None = None
    # Which overlapping areas to look for, so competitor extraction is targeted
    # rather than a generic summary of the peer.
    overlap_focus: list[str] = Field(default_factory=list)
    # 'disclosed' when the subject's own filing names this peer, 'industry'
    # otherwise. Surfaced in the report so a reader can tell evidenced peer
    # relationships from analyst-chosen comparison sets.
    basis: str = "industry"


class SourcesSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # An 8-K's primary document is a cover page; the substance is Exhibit 99.1.
    # Without this, earnings filings cost retrieval and contribute nothing.
    fetch_earnings_releases: bool = True
    # Cap on how many earnings releases to mine, newest first.
    max_earnings_releases: int = 2
    # Competitor first-party filings, used ONLY for competitive-overlap evidence.
    fetch_competitor_filings: bool = False
    max_competitor_documents: int = 3
    # Char budget for competitor text, kept separate from the primary budget so
    # competitor context can never crowd out the subject company's own evidence.
    competitor_char_budget: int = 15_000
    # Peers declared per subject company. A single global pool is wrong: it would
    # analyse a memory maker against equipment vendors.
    peer_map: dict[str, list[CompetitorSourceConfig]] = Field(default_factory=dict)

    def peers_for(self, company_name: str) -> list[CompetitorSourceConfig]:
        """Peers configured for a subject company, matched leniently on name.

        Falls back to an empty list rather than a global pool: analysing a company
        against unrelated peers produces a confidently wrong competitive section,
        so no peers is the safer default.
        """
        def norm(n: str) -> str:
            return " ".join(
                "".join(ch.lower() if ch.isalnum() else " " for ch in n).split()
            )

        target = norm(company_name)
        for subject, peers in self.peer_map.items():
            subject_norm = norm(subject)
            if target == subject_norm or target.startswith(subject_norm) or subject_norm.startswith(target):
                return peers
        return []


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_agent_contact: str = "anonymous@example.com"
    http: HttpSettings = Field(default_factory=HttpSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    filings: FilingSettings = Field(default_factory=FilingSettings)
    sources: SourcesSettings = Field(default_factory=SourcesSettings)
    financials: FinancialSettings = Field(default_factory=FinancialSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)

    @property
    def user_agent(self) -> str:
        """SEC requires a descriptive User-Agent with contact information.

        Requests without one are rejected (HTTP 403), which is exactly the
        failure we observed when probing with a bare client.
        """
        return f"SemiconductorCompanyIntelligenceAgent/0.1 ({self.user_agent_contact})"

    @property
    def cache_root(self) -> Path:
        return project_root() / self.cache.root


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or (project_root() / "config" / "settings.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)


# --------------------------------------------------------------------------- #
# companies.yaml
# --------------------------------------------------------------------------- #


class CompanyConfig(BaseModel):
    """One company's registration entry.

    Adding a company to the supported set means adding one of these to YAML.
    No code change is required, which is the extensibility guarantee.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    ticker: str | None = None
    cik: str | None = None
    aliases: list[str] = Field(default_factory=list)
    sector_role: str | None = None
    fiscal_year_end: str | None = None
    relevant_process_steps: list[str] = Field(default_factory=list)

    @property
    def padded_cik(self) -> str | None:
        """SEC APIs want a zero-padded 10-digit CIK."""
        if not self.cik:
            return None
        return str(self.cik).lstrip("0").zfill(10)


class CompanyRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    companies: list[CompanyConfig] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        out: list[str] = []
        for c in self.companies:
            out.append(c.name)
            out.extend(c.aliases)
        return out


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> CompanyRegistry:
    cfg_path = path or (project_root() / "config" / "companies.yaml")
    if not cfg_path.exists():
        return CompanyRegistry()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return CompanyRegistry.model_validate(raw)


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #

API_KEY_ENV_VAR = "OPENAI_API_KEY"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a ``.env`` file into ``os.environ``.

    Implemented here rather than by adding ``python-dotenv``: the format we need
    is a handful of lines, and a dependency for that is not worth the supply-chain
    surface in a portfolio project.

    Deliberately conservative:

    * Real environment variables win. A shell export is an explicit act by the
      operator and must not be silently overridden by a checked-out file.
    * Missing file is not an error - steps 1-2 run without any secrets at all.
    * Only simple ``KEY=VALUE`` lines are parsed. No interpolation, no command
      substitution, because those are exactly the features that turn a config file
      into an execution vector.

    Returns the pairs that were actually set.
    """
    env_path = path or (project_root() / ".env")
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    try:
        raw = env_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one layer of matching quotes, which is how most people paste keys.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not key:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        loaded[key] = value

    return loaded


def get_api_key() -> str | None:
    """Return the OpenAI key if present, loading ``.env`` as a fallback.

    Deliberately lazy: steps 1-2 are fully functional without any API key, and
    importing this module must never require one. The ``.env`` file is only read
    when a key is actually being looked up, so a retrieval-only run never touches
    it.
    """
    key = os.environ.get(API_KEY_ENV_VAR)
    if key:
        return key
    load_dotenv()
    return os.environ.get(API_KEY_ENV_VAR)
