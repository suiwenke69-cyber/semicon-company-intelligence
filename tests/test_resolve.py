"""Tests for company resolution and the config-driven extensibility guarantee.

Resolution correctness is a correctness-of-subject issue: resolving "Applied
Materials" to the wrong CIK would produce a fluent, fully-cited report about the
wrong company.
"""

from __future__ import annotations

import pytest

from semicon.config import CompanyRegistry, load_registry
from semicon.resolve import (
    ResolutionError,
    TickerRecord,
    _normalise,
    _similarity,
    resolve_company,
)


class FakeHttpClient:
    """Stands in for HttpClient so resolution tests never touch the network."""

    def __init__(self, records: list[TickerRecord]):
        self._payload = {
            str(i): {"cik_str": r.cik, "ticker": r.ticker, "title": r.title}
            for i, r in enumerate(records)
        }

    def get_json(self, url: str, **kwargs: object) -> object:
        return self._payload


@pytest.fixture
def client() -> FakeHttpClient:
    return FakeHttpClient(
        [
            TickerRecord(cik=6951, ticker="AMAT", title="APPLIED MATERIALS INC /DE"),
            TickerRecord(cik=707549, ticker="LRCX", title="LAM RESEARCH CORP"),
            TickerRecord(cik=319201, ticker="KLAC", title="KLA CORP"),
            TickerRecord(cik=937966, ticker="ASML", title="ASML HOLDING NV"),
            TickerRecord(cik=723125, ticker="MU", title="MICRON TECHNOLOGY INC"),
            TickerRecord(cik=1045810, ticker="NVDA", title="NVIDIA CORP"),
        ]
    )


class TestNormalisation:
    def test_legal_suffixes_stripped(self):
        assert _normalise("Applied Materials, Inc.") == "applied materials"
        assert _normalise("APPLIED MATERIALS INC /DE") == "applied materials"

    def test_similarity_ignores_suffixes(self):
        assert _similarity("Applied Materials", "APPLIED MATERIALS INC /DE") == 1.0


class TestResolution:
    def test_config_cik_wins_without_network(self):
        """Applied Materials is pinned in companies.yaml, so resolution needs no
        ticker-map lookup at all."""
        settings = _settings()
        result = resolve_company("Applied Materials", client=None, settings=settings)  # type: ignore[arg-type]
        assert result.cik == "0000006951"
        assert result.matched_by == "config"
        assert result.match_confidence == 1.0

    def test_ticker_lookup(self, client):
        result = resolve_company("LRCX", client, _settings())
        assert result.cik == "0000707549"
        assert result.matched_by == "ticker"

    def test_name_lookup(self, client):
        result = resolve_company("KLA Corp", client, _settings())
        assert result.cik == "0000319201"

    def test_fuzzy_match_reports_candidates(self, client):
        result = resolve_company("Micron Tecnology", client, _settings())
        assert result.cik == "0000723125"
        assert result.matched_by == "fuzzy"
        assert result.fuzzy_candidates

    def test_fuzzy_rejected_when_disallowed(self, client):
        with pytest.raises(ResolutionError):
            resolve_company(
                "Micron Tecnology", client, _settings(), allow_fuzzy=False
            )

    def test_nonsense_query_refuses_rather_than_guessing(self, client):
        """A private company or typo must fail loudly, never silently resolve."""
        with pytest.raises(ResolutionError):
            resolve_company("Zzzz Nonexistent Semiconductor", client, _settings())


class TestRegistryExtensibility:
    """Adding a company must be a YAML edit, not a code change."""

    def test_all_target_companies_registered(self):
        registry = load_registry()
        names = {c.name for c in registry.companies}
        for expected in (
            "Applied Materials", "Lam Research", "KLA", "ASML",
            "Micron", "NVIDIA", "AMD", "Infineon", "NXP", "Analog Devices",
        ):
            assert expected in names

    def test_applied_materials_is_fully_pinned(self):
        registry = load_registry()
        amat = next(c for c in registry.companies if c.ticker == "AMAT")
        assert amat.cik == "0000006951"
        assert amat.padded_cik == "0000006951"
        assert amat.sector_role == "equipment"
        assert "Deposition" in amat.relevant_process_steps

    def test_cik_padding(self):
        registry = CompanyRegistry.model_validate(
            {"companies": [{"name": "X", "cik": "6951"}]}
        )
        assert registry.companies[0].padded_cik == "0000006951"

    def test_foreign_private_issuer_supported_by_config(self):
        """ASML files 20-F, not 10-K. Configuration must express that without
        any code special-casing."""
        from semicon.config import load_settings

        settings = load_settings()
        assert "20-F" in settings.filings.annual_forms
        assert "40-F" in settings.filings.annual_forms

    def test_unknown_company_gets_synthesised_config(self, client):
        """An unregistered registrant still works, just without curated hints."""
        result = resolve_company("NVIDIA Corp", client, _settings())
        assert result.config.name
        assert result.cik


def _settings():
    from semicon.config import load_settings

    return load_settings()
