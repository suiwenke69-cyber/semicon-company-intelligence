"""Regression tests for deterministic table parsing.

Every case here corresponds to a real defect found while parsing AMAT's actual
10-Ks. They run fully offline and cost nothing.
"""

from __future__ import annotations

import pytest

from semicon.sources.sec_tables import (
    _parse_row,
    extract_revenue_breakdown,
    parse_geographic_table,
    parse_segment_table,
)
from tests.fixtures import (
    GEO_FY2023,
    GEO_FY2025,
    NON_DATA_ROWS,
    SEGMENT_FY2024,
    SEGMENT_FY2025,
)


class TestRowParsing:
    """Row tokenisation, which is where most of the subtlety lives."""

    def test_row_without_thousands_separator(self):
        """'Europe 962 3% ...' - 962 has no comma and must not fuse with the 3."""
        label, amounts, pcts = _parse_row(
            "Europe 962 3% 1,443 5% (33) %"
        )
        assert label == "Europe"
        assert amounts == [962.0, 1443.0]
        assert pcts == [3.0, 5.0, -33.0]

    def test_parenthesised_negative_change(self):
        """'(16) %' has a space, so a naive tokeniser reads 16 as money."""
        label, amounts, pcts = _parse_row(
            "China $ 8,529 30% $ 10,117 37% (16) %"
        )
        assert label == "China"
        assert amounts == [8529.0, 10117.0]
        assert -16.0 in pcts

    def test_dollar_signs_do_not_pollute_label(self):
        label, _, _ = _parse_row(
            "Semiconductor Systems $ 20,798 73% $ 19,911 73% 4 %"
        )
        assert label == "Semiconductor Systems"
        assert "$" not in label

    def test_small_segment_values_are_accepted(self):
        """A magnitude filter wrongly discarded AMAT's real $155M segment."""
        label, amounts, _ = _parse_row("Corporate and Other 155 1% 219 1% (29) %")
        assert label == "Corporate and Other"
        assert amounts[0] == 155.0

    def test_trailing_dash_stripped(self):
        """FY2023 renders the no-change cell as a bare '-' inside the label."""
        from semicon.sources.sec_tables import _LABEL_JUNK

        label, _, _ = _parse_row(
            "China $ 7,247 27% $ 7,254 28% $ 7,535 33% - % (4) %"
        )
        assert _LABEL_JUNK.sub("", label).strip() == "China"

    @pytest.mark.parametrize("row", NON_DATA_ROWS)
    def test_non_data_rows_rejected(self, row):
        assert _parse_row(row) is None

    def test_percentage_only_row_rejected(self):
        """A market-mix table ('Foundry, logic and other 68 % 77 %') is not revenue."""
        assert _parse_row("Foundry, logic and other 68 % 77 %") is None


class TestSegmentTable:
    def test_fy2025_segments(self):
        segments = parse_segment_table(SEGMENT_FY2025, "sec_10_k_2025", 2025)
        names = [s.name for s in segments]
        assert names == [
            "Semiconductor Systems",
            "Applied Global Services",
            "Corporate and Other",
        ]
        top = segments[0]
        assert top.revenue_millions == 20798.0
        assert top.revenue_pct_of_total == 73.0
        assert top.revenue_prior_year_millions == 19911.0

    def test_fy2024_retains_small_segments(self):
        """Display ($885M) and Corporate and Other ($155M) must survive."""
        segments = parse_segment_table(SEGMENT_FY2024, "sec_10_k_2024", 2024)
        names = [s.name for s in segments]
        assert "Display" in names
        assert "Corporate and Other" in names

    def test_segments_sum_to_reported_total(self):
        segments = parse_segment_table(SEGMENT_FY2025, "sec_10_k_2025", 2025)
        assert sum(s.revenue_millions for s in segments) == 28368.0

    def test_change_pct_computed(self):
        segments = parse_segment_table(SEGMENT_FY2025, "sec_10_k_2025", 2025)
        assert segments[0].revenue_change_pct == pytest.approx(4.5, abs=0.1)


class TestGeographicTable:
    def test_fy2025_regions(self):
        regions, total = parse_geographic_table(GEO_FY2025, "sec_10_k_2025", 2025)
        assert total == 28368.0
        labels = [r.region for r in regions]
        assert "China" in labels
        assert "Europe" in labels
        assert len(regions) == 8

    def test_subtotal_flagged(self):
        regions, _ = parse_geographic_table(GEO_FY2025, "sec_10_k_2025", 2025)
        apac = next(r for r in regions if r.region == "Asia Pacific")
        assert apac.is_subtotal is True

    def test_constituent_regions_reconcile_to_total(self):
        """Subtotals must be excluded or regions double-count."""
        regions, total = parse_geographic_table(GEO_FY2025, "sec_10_k_2025", 2025)
        constituent = sum(r.revenue_millions for r in regions if not r.is_subtotal)
        assert constituent == total

    def test_three_year_layout_with_different_wording(self):
        """FY2023 says 'Net sales ... were as follows' and has 3 year columns."""
        regions, total = parse_geographic_table(GEO_FY2023, "sec_10_k_2023", 2023)
        assert total == 26517.0
        china = next(r for r in regions if r.region == "China")
        assert china.revenue_millions == 7247.0
        assert china.prior_year_millions == 7254.0
        constituent = sum(r.revenue_millions for r in regions if not r.is_subtotal)
        assert constituent == total


class TestBreakdownIntegration:
    def test_combined_extraction(self):
        """A filing containing both tables must yield both, with no warnings."""
        result = extract_revenue_breakdown(
            SEGMENT_FY2025 + "\n" + GEO_FY2025, "sec_10_k_2025", 2025
        )
        assert result.total_revenue_millions == 28368.0
        assert len(result.by_segment) == 3
        assert len(result.by_geography) == 8
        assert result.warnings == []

    def test_geographic_only_still_warns_about_missing_segments(self):
        result = extract_revenue_breakdown(GEO_FY2025, "sec_10_k_2025", 2025)
        assert result.total_revenue_millions == 28368.0
        assert any("segment" in w.lower() for w in result.warnings)

    def test_missing_geographic_table_warns(self):
        """A filing with no such table must warn, never silently invent."""
        result = extract_revenue_breakdown(
            "This filing contains no revenue tables at all.", "x", 2025
        )
        assert result.by_geography == []
        assert any("geographic" in w.lower() for w in result.warnings)


class TestBacklogIsNotRevenue:
    """Defect: an over-broad lead-in regex parsed the BACKLOG table as segment revenue.

    AMAT's backlog table is introduced by a sentence that contains both "revenue"
    and "segment", so a loose pattern matched it. The result was a segment table
    showing $7,105M for Semiconductor Systems instead of the true $20,798M - every
    segment figure silently wrong, while still passing a parts-sum-to-total check
    because the backlog table has a total row of its own.
    """

    def test_backlog_table_not_parsed_as_segment_revenue(self):
        from tests.fixtures import BACKLOG_SEGMENT

        segments = parse_segment_table(BACKLOG_SEGMENT, "sec_10_k_2025", 2025)
        # Either nothing is returned, or nothing that looks like AMAT's real
        # segment revenue. It must never claim $7,105M for Semiconductor Systems.
        assert all(s.revenue_millions != 7105.0 for s in segments), (
            "backlog figures were reported as segment revenue"
        )

    def test_loose_preamble_alone_yields_nothing(self):
        from tests.fixtures import BACKLOG_PREAMBLE

        segments = parse_segment_table(BACKLOG_PREAMBLE, "sec_10_k_2025", 2025)
        assert segments == []

    def test_real_segment_table_still_parses(self):
        """The fix must not have broken the genuine table."""
        segments = parse_segment_table(SEGMENT_FY2025, "sec_10_k_2025", 2025)
        assert segments[0].name == "Semiconductor Systems"
        assert segments[0].revenue_millions == 20798.0


class TestTableUnitScaling:
    """Defect: table units were assumed to be millions.

    KLA reports its geographic revenue in THOUSANDS in one table and millions in
    another. Assuming millions rendered "North America 1,757,337M" for a company
    whose total revenue is about $13.6 billion - wrong by a factor of 1000, and
    displayed with exactly the same confidence as a correct figure.

    The unit is now read from the table's own header. A confident number that is
    wrong by 1000x is worse than a missing one, so these tests are deliberately
    strict.
    """

    # Realistic shape: year header, unit declaration, then rows carrying amounts
    # AND percentages, as every actual filing table does.
    THOUSANDS_TABLE = """The following table presents our revenue disaggregated by geographic region:

Change
2026 2025 2026 over 2025

(In thousands, except percentages)
China $ 4,048,000 53% $ 3,800,000 54% 7 %
Taiwan 3,644,000 47% 3,200,000 46% 14 %
Total $ 7,692,000 100% $ 7,000,000 100% 10 %
"""

    MILLIONS_TABLE = """Net revenue by geographic region was as follows:

Change
2026 2025 2026 over 2025

(In millions, except percentages)
China $ 4,048 53% $ 3,800 54% 7 %
Taiwan 3,644 47% 3,200 46% 14 %
Total $ 7,692 100% $ 7,000 100% 10 %
"""

    def test_thousands_scaled_to_millions(self):
        regions, total = parse_geographic_table(
            self.THOUSANDS_TABLE, "sec_10_k_2026", 2026
        )
        china = next(r for r in regions if r.region == "China")
        assert china.revenue_millions == 4048.0, "thousands were not scaled"
        assert total == 7692.0

    def test_millions_left_alone(self):
        regions, total = parse_geographic_table(
            self.MILLIONS_TABLE, "sec_10_k_2026", 2026
        )
        china = next(r for r in regions if r.region == "China")
        assert china.revenue_millions == 4048.0
        assert total == 7692.0

    def test_billions_scaled_up(self):
        text = self.MILLIONS_TABLE.replace("millions", "billions")
        regions, total = parse_geographic_table(text, "x", 2026)
        china = next(r for r in regions if r.region == "China")
        assert china.revenue_millions == 4_048_000.0

    def test_unit_variants_detected(self):
        from semicon.sources.sec_tables import _detect_unit_scale

        for header, expected in (
            ("(In thousands, except percentages)", 0.001),
            ("(in millions)", 1.0),
            ("Revenue (in millions)", 1.0),
            ("(In  thousands)", 0.001),
            ("(In billions)", 1000.0),
            ("no unit stated", 1.0),
        ):
            got = _detect_unit_scale("pad " * 10 + header + " tail " * 30, 0)
            assert got == expected, f"{header!r} -> {got}, expected {expected}"

    def test_segment_table_also_scaled(self):
        segments = parse_segment_table(self.THOUSANDS_TABLE, "x", 2026)
        # China/Taiwan are not segment names, so this may parse nothing; the point
        # is that no unscaled millions figure leaks through.
        assert all(
            (s.revenue_millions is None) or s.revenue_millions < 100_000
            for s in segments
        )
