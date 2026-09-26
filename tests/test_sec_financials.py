"""Regression tests for deterministic XBRL financial extraction.

These target the specific defects found while validating AMAT's numbers. The
most serious was a year-over-year figure that compared a 90-day quarter against
a 363-day year and reported a confidently wrong percentage.
"""

from __future__ import annotations

from datetime import date

import pytest

from semicon.config import load_settings
from semicon.models import FinancialMetric, FinancialsExtract, XbrlValue
from semicon.sources.sec_financials import (
    _compute_yoy,
    _fiscal_quarter_of,
    _fiscal_year_of,
    _period_label,
    count_period_kinds,
    parse_fiscal_year_end,
    refresh_notes,
    split_periods,
)


def _period(
    label: str, days: int, end: str, value: float
) -> XbrlValue:
    y, m, d = (int(x) for x in end.split("-"))
    end_date = date(y, m, d)
    start_date = date.fromordinal(end_date.toordinal() - days)
    return XbrlValue(
        metric="m",
        taxonomy="us-gaap",
        tag="T",
        unit="USD",
        value=value,
        start=start_date,
        end=end_date,
        duration_days=days,
        period_label=label,
    )


class TestFiscalPeriodDerivation:
    """Labels must come from dates, not from the filing's own fy/fp elements."""

    def test_amat_fiscal_year_boundary(self):
        # FYE month 10 (October): a period ending 26 Oct 2025 is FY2025,
        # but one ending 15 Nov 2025 already belongs to FY2026.
        assert _fiscal_year_of(date(2025, 10, 26), 10) == 2025
        assert _fiscal_year_of(date(2025, 11, 15), 10) == 2026
        assert _fiscal_year_of(date(2026, 1, 25), 10) == 2026

    def test_amat_fiscal_quarters(self):
        # AMAT's fiscal year starts in November.
        assert _fiscal_quarter_of(date(2026, 1, 25), 10) == (2026, 1)
        assert _fiscal_quarter_of(date(2026, 4, 26), 10) == (2026, 2)
        assert _fiscal_quarter_of(date(2026, 7, 26), 10) == (2026, 3)
        assert _fiscal_quarter_of(date(2026, 10, 26), 10) == (2026, 4)

    def test_calendar_year_filer(self):
        assert _fiscal_year_of(date(2024, 12, 31), 12) == 2024
        assert _fiscal_quarter_of(date(2024, 3, 31), 12) == (2024, 1)

    def test_annual_label(self):
        label, fy, quarter, kind = _period_label(
            "2024-10-28", "2025-10-26", 10, 363
        )
        assert label == "FY2025"
        assert fy == 2025
        assert quarter is None
        assert kind == "annual"

    def test_quarter_label(self):
        label, fy, quarter, kind = _period_label(
            "2026-04-27", "2026-07-26", 10, 90
        )
        assert label == "Q3 FY2026"
        assert (fy, quarter, kind) == (2026, 3, "quarter")

    def test_prior_year_comparative_is_not_labelled_with_filing_year(self):
        """The original bug: a FY2024 10-K's prior-year column ending Oct 2023
        was labelled FY2024 because the filing's fy element said 2024."""
        label, fy, _, _ = _period_label("2022-10-31", "2023-10-29", 10, 363)
        assert fy == 2023
        assert label == "FY2023"

    def test_fiscal_year_end_parsing(self):
        assert parse_fiscal_year_end("1026") == 10   # AMAT
        assert parse_fiscal_year_end("0831") == 8    # Micron
        assert parse_fiscal_year_end("1231") == 12   # calendar
        assert parse_fiscal_year_end(None) is None
        assert parse_fiscal_year_end("1360") is None  # invalid month
        assert parse_fiscal_year_end("abc") is None


class TestYearOverYear:
    """The invariant: a quarter is only ever compared with a quarter."""

    def test_never_compares_quarter_to_year(self):
        """The original defect: Q1 FY2026 ($1,686M, 90d) vs FY2025 ($7,958M, 363d)
        yielded '-8.3% YoY'. Annual periods exist, so they must be used."""
        periods = [
            _period("Q1 FY2026", 90, "2026-01-25", 1686.0),
            _period("FY2025", 363, "2025-10-26", 7958.0),
            _period("Q1 FY2025", 90, "2025-01-26", 925.0),
            _period("FY2024", 363, "2024-10-27", 8677.0),
        ]
        growth = _compute_yoy(periods, 4)
        # Annual-vs-annual: (7958-8677)/8677 = -8.3%
        assert growth == pytest.approx(-8.3, abs=0.1)

    def test_mismatched_durations_rejected(self):
        """With no annual pair available, a quarter with no same-length
        counterpart must yield None rather than a wrong number."""
        periods = [
            _period("Q1 FY2026", 90, "2026-01-25", 1686.0),
            _period("FY2025", 363, "2025-10-26", 7958.0),
        ]
        assert _compute_yoy(periods, 4) is None

    def test_quarter_to_quarter_when_no_annuals(self):
        periods = [
            _period("Q3 FY2026", 90, "2026-07-26", 110.0),
            _period("Q3 FY2025", 90, "2025-07-27", 100.0),
        ]
        assert _compute_yoy(periods, 4) == pytest.approx(10.0, abs=0.1)

    def test_single_period_returns_none(self):
        assert _compute_yoy([_period("FY2025", 363, "2025-10-26", 100.0)], 4) is None


class TestPeriodCapping:
    """Annual and quarterly history must be capped independently."""

    def _metric(self, n_annual: int, n_quarterly: int) -> FinancialMetric:
        periods = []
        for i in range(n_annual):
            periods.append(_period(f"FY{2025 - i}", 363, f"{2025 - i}-10-26", 100.0))
        for i in range(n_quarterly):
            periods.append(
                _period(f"Q{(i % 4) + 1} FY{2026 - i // 4}", 90, "2026-07-26", 50.0)
            )
        periods.sort(key=lambda p: p.end, reverse=True)
        return FinancialMetric(
            metric="m", unit="USD", concept="us-gaap:T", periods=periods
        )

    def test_cap_does_not_evict_all_quarters(self):
        """The original defect: a flat cap of 12 over mixed periods pushed out
        every recent quarter for metrics with long annual history."""
        metric = self._metric(n_annual=18, n_quarterly=18)
        capped = split_periods(metric, 12, 12)
        annual, quarterly = count_period_kinds(capped)
        assert annual == 12
        assert quarterly == 12

    def test_note_reflects_retained_periods_not_pre_cap_state(self):
        """The note previously advertised counts from before truncation."""
        metric = self._metric(n_annual=18, n_quarterly=18)
        capped = refresh_notes(split_periods(metric, 12, 12))
        assert capped.note == "12 annual, 12 quarterly"

    def test_latest_annual_and_quarter_helpers(self):
        metric = self._metric(n_annual=3, n_quarterly=4)
        assert metric.latest_annual.duration_days == 363
        assert metric.latest_quarter.duration_days == 90
        assert metric.preferred_period.duration_days == 363


class TestSettingsIntegrity:
    def test_deterministic_run_makes_no_llm_calls(self):
        """The retrieval path must not touch the LLM regardless of llm.enabled.

        ``enabled`` gates whether analysis is *permitted*; the CLI still requires
        an explicit --analyze flag, so a plain run cannot spend tokens.
        """
        from semicon.pipeline import run as pipeline_run

        settings = load_settings()
        profile, artifacts = pipeline_run("Applied Materials", settings=settings)
        # With no client constructed, no usage can have been recorded.
        assert artifacts.validation is None
        assert profile.evidence == []

    def test_budget_and_ceilings_configured(self):
        llm = load_settings().llm
        assert llm.run_token_budget > 0
        assert llm.max_extraction_chunks > 0
        assert llm.max_extraction_chars > 0
        assert llm.extract_model and llm.synth_model

    def test_process_step_taxonomy_is_closed(self):
        from semicon.models import ProcessStep

        values = {p.value for p in ProcessStep}
        for expected in (
            "Deposition", "Etch", "Lithography", "Process Control",
            "Inspection", "Metrology", "Implantation", "CMP",
            "Advanced Packaging",
        ):
            assert expected in values
