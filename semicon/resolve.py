"""Company name -> CIK resolution.

Resolution order, cheapest first:

1. An explicit ``cik`` in ``companies.yaml`` (no network, zero ambiguity).
2. Exact ticker match against SEC's official ticker map.
3. Exact/alias name match against the ticker map.
4. Fuzzy match, which is *reported as ambiguous* rather than silently accepted.

The governing principle: never silently resolve to the wrong legal entity. Two
different issuers can share a ticker across exchanges, and a misspelling should
surface as a question, not as a confident report about the wrong company.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import CompanyConfig, CompanyRegistry, Settings, load_registry
from .http import HttpClient, RetrievalError

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class ResolutionError(ValueError):
    """Raised when a company cannot be resolved with acceptable confidence."""


@dataclass
class TickerRecord:
    cik: int
    ticker: str
    title: str

    @property
    def padded_cik(self) -> str:
        return str(self.cik).zfill(10)


@dataclass
class ResolvedCompany:
    """A company we are confident enough about to retrieve filings for."""

    name: str
    cik: str                       # zero-padded, SEC-ready
    ticker: str | None
    sec_title: str | None
    matched_by: str                # 'config' | 'ticker' | 'alias' | 'name'
    match_confidence: float        # 0.0-1.0
    config: CompanyConfig
    fuzzy_candidates: list[str] | None = None

    @property
    def is_ambiguous(self) -> bool:
        return self.match_confidence < 0.95


def load_ticker_map(client: HttpClient) -> list[TickerRecord]:
    """Fetch and normalise SEC's official ticker->CIK map.

    This is a single ~800 KB file listing every SEC registrant, so it is cached
    aggressively and reused for all companies in a run.
    """
    try:
        raw = client.get_json(SEC_TICKER_MAP_URL)
    except RetrievalError as exc:
        raise ResolutionError(f"Could not load SEC ticker map: {exc}") from exc

    records: list[TickerRecord] = []
    if isinstance(raw, dict):
        # Format: {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}, ...}
        for item in raw.values():
            if isinstance(item, dict) and "cik_str" in item:
                records.append(
                    TickerRecord(
                        cik=int(item["cik_str"]),
                        ticker=str(item.get("ticker", "")).upper(),
                        title=str(item.get("title", "")),
                    )
                )
    return records


def _normalise(s: str) -> str:
    """Fold a company name for comparison.

    Drops *legal designators* only: 'Inc', 'Corp', 'NV', 'PLC' and the state-of-
    incorporation suffix EDGAR appends ('APPLIED MATERIALS INC /DE'). So
    'Applied Materials, Inc.' and 'APPLIED MATERIALS INC /DE' compare as equal.

    Deliberately NOT dropped: semantically meaningful words such as
    'Technology', 'Technologies' and 'Holdings'. An earlier version stripped
    ' technologies', which collapsed 'Micron Technology' to 'micron' and caused
    the fuzzy matcher to reject the near-miss 'Micron Tecnology' with a
    similarity of 0.545. Words that carry identity must survive normalisation.
    """
    s = s.lower()
    # State-of-incorporation suffix, e.g. ' /de' or ' /md'.
    s = re.sub(r"\s*/\s*[a-z]{2}\b", " ", s)
    for token in (
        " inc", " corp", " corporation", " company", " co", " ltd", " limited",
        " plc", " nv", " se", " ag", " sa", " llc", " lp", " gmbh",
        ",", ".", "-", "&",
    ):
        s = s.replace(token, " ")
    return " ".join(s.split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def resolve_company(
    query: str,
    client: HttpClient,
    settings: Settings,
    registry: CompanyRegistry | None = None,
    *,
    allow_fuzzy: bool = True,
    fuzzy_threshold: float = 0.72,
) -> ResolvedCompany:
    """Resolve a user-supplied company name to a CIK.

    Args:
        query: e.g. "Applied Materials".
        allow_fuzzy: when False, a fuzzy-only match raises instead of returning.
        fuzzy_threshold: below this ratio we refuse rather than guess.

    Raises:
        ResolutionError: if resolution fails or the best match is ambiguous.
    """
    registry = registry or load_registry()
    q = query.strip()
    q_norm = _normalise(q)

    # 1. Explicit CIK in config: authoritative, no network needed.
    for cfg in registry.companies:
        if cfg.cik and (
            q_norm == _normalise(cfg.name)
            or (cfg.ticker and q.upper() == cfg.ticker.upper())
            or any(q_norm == _normalise(a) for a in cfg.aliases)
        ):
            return ResolvedCompany(
                name=cfg.name,
                cik=cfg.padded_cik or "",
                ticker=cfg.ticker,
                sec_title=cfg.name,
                matched_by="config",
                match_confidence=1.0,
                config=cfg,
            )

    records = load_ticker_map(client)

    # 2. Exact ticker match.
    for rec in records:
        if rec.ticker and q.upper() == rec.ticker:
            return ResolvedCompany(
                name=rec.title,
                cik=rec.padded_cik,
                ticker=rec.ticker,
                sec_title=rec.title,
                matched_by="ticker",
                match_confidence=1.0,
                config=_config_for(rec, registry),
            )

    # 3. Alias / exact normalised name match, with config aliases given priority
    #    because they encode human intent for names SEC spells differently.
    for cfg in registry.companies:
        for candidate in [cfg.name, *cfg.aliases]:
            if _normalise(candidate) == q_norm and cfg.ticker:
                for rec in records:
                    if rec.ticker.upper() == cfg.ticker.upper():
                        return ResolvedCompany(
                            name=cfg.name,
                            cik=rec.padded_cik,
                            ticker=rec.ticker,
                            sec_title=rec.title,
                            matched_by="alias",
                            match_confidence=0.99,
                            config=cfg,
                        )

    for rec in records:
        if _normalise(rec.title) == q_norm:
            return ResolvedCompany(
                name=rec.title,
                cik=rec.padded_cik,
                ticker=rec.ticker,
                sec_title=rec.title,
                matched_by="name",
                match_confidence=1.0,
                config=_config_for(rec, registry),
            )

    # 4. Fuzzy. Scored, and surfaced with its candidates.
    if not allow_fuzzy:
        raise ResolutionError(
            f"'{query}' did not match any SEC registrant exactly. "
            f"Add it to config/companies.yaml or check the spelling."
        )

    scored = sorted(
        ((_similarity(q, rec.title), rec) for rec in records),
        key=lambda t: t[0],
        reverse=True,
    )
    best_score, best = scored[0] if scored else (0.0, None)

    if best is None or best_score < fuzzy_threshold:
        raise ResolutionError(
            f"Could not confidently resolve '{query}' (best similarity "
            f"{best_score:.2f} < {fuzzy_threshold}). It may be a private company, "
            f"a foreign non-filer, or misspelled. Add an explicit entry to "
            f"config/companies.yaml to proceed."
        )

    return ResolvedCompany(
        name=best.title,
        cik=best.padded_cik,
        ticker=best.ticker,
        sec_title=best.title,
        matched_by="fuzzy",
        match_confidence=round(best_score, 3),
        config=_config_for(best, registry),
        fuzzy_candidates=[r.title for _, r in scored[:5]],
    )


def _config_for(rec: TickerRecord, registry: CompanyRegistry) -> CompanyConfig:
    """Attach the matching YAML entry if one exists, else synthesise a minimal one.

    Synthesising means an unregistered company still works end to end; it simply
    lacks the curated aliases and process-step hints.
    """
    for cfg in registry.companies:
        if cfg.ticker and cfg.ticker.upper() == rec.ticker.upper():
            return cfg
        if cfg.cik and str(cfg.cik).lstrip("0") == str(rec.cik):
            return cfg
    return CompanyConfig(name=rec.title, ticker=rec.ticker, cik=str(rec.cik))
