"""Tests for chunking, extraction gating, analysis coercion and validation.

These cover the LLM-stage logic that can be verified without an API key. The
central guarantee under test is the anti-hallucination gate: a claim whose
supporting quote is not present verbatim in the source must be discarded.
"""

from __future__ import annotations

import pytest

from semicon.analyze import _coerce_process_steps, _coerce_evidence, build_ledger_text
from semicon.chunk import (
    Chunk,
    Section,
    chunk_section,
    dedupe_chunks,
    find_sections,
    plan_extraction,
    select_sections,
    score_chunk,
)
from semicon.extract import _claim_id, _normalise_quote, _verify_quote
from semicon.llm.client import LlmError, extract_json
from semicon.models import (
    ClaimType,
    CompanyIdentity,
    CompanyProfile,
    CustomerDisclosure,
    CustomerInsight,
    Evidence,
    ProcessStep,
    ProductItem,
    Source,
    SourceKind,
)
from semicon.validate import validate_profile
from tests.fixtures import GEO_FY2025, SEGMENT_FY2025

# A miniature filing: a table-of-contents block followed by real section bodies.
# The TOC entries are deliberately short, which is exactly how the real documents
# look and how they are told apart from the body. Body sections are long enough to
# clear the 400-character threshold that filters contents entries out.
SAMPLE_10K = """Item 1: Business
3
Item 1A: Risk Factors
9
Item 7: Management's Discussion and Analysis
18

Item 1: Business
Applied Materials, Inc. is a leader in materials engineering solutions used to produce semiconductors. We operate in two reportable segments: Semiconductor Systems and Applied Global Services. Our deposition, etch, ion implantation and chemical mechanical planarization products serve leading-edge logic, dynamic random-access memory and NAND customers around the world. We also provide spares, services and software upgrades that extend the productivity of installed equipment.

Item 1A: Risk Factors
Our business depends on capital spending by semiconductor manufacturers. A downturn in memory or logic investment would reduce demand for our equipment and could materially harm our results of operations and financial condition. We also face intense competition from other equipment suppliers. Customers may delay or cancel orders during periods of economic uncertainty, and a small number of large customers account for a significant share of our revenue.

Item 7: Management's Discussion and Analysis
Net revenue increased in fiscal 2025, driven by demand for advanced packaging equipment used in artificial intelligence accelerators. We expect continued investment in leading-edge capacity, although the timing of customer spending remains uncertain. The memory market has historically been cyclical, and our results have varied with it. Research and development spending rose as we invested in next-generation deposition and etch platforms.
"""


class TestSectionDiscovery:
    def test_toc_entries_filtered_out(self):
        """Every heading appears twice; only the body occurrence is a section."""
        sections = find_sections(SAMPLE_10K)
        # Business must be the long body copy, not the 20-char contents line.
        business = next(s for s in sections if s.item == "1")
        assert business.chars > 200
        assert "materials engineering solutions" in business.text

    def test_colon_separator_supported(self):
        """AMAT writes 'Item 1: Business' where most filers write 'Item 1.'"""
        assert find_sections(SAMPLE_10K)

    def test_period_separator_also_supported(self):
        """The conventional 'Item 1. Business' form must work for other filers."""
        text = SAMPLE_10K.replace("Item 1: Business", "Item 1. Business").replace(
            "Item 7: Management's", "Item 7. Management's"
        )
        assert find_sections(text)

    def test_missing_sections_returns_empty(self):
        assert find_sections("no items here at all") == []

    def test_priority_order_follows_declaration(self):
        """Section order is declared, not alphabetical.

        Business (Item 1) is the richest source of product and competitor detail,
        so it is processed before Risk Factors (Item 1A) even though '1A' would
        sort adjacent to '1'.
        """
        sections = select_sections(find_sections(SAMPLE_10K))
        items = [s.item for s in sections]
        assert items.index("1") < items.index("1A")

    def test_only_relevant_items_kept(self):
        """Item 4 (Mine Safety) is not part of the framework and must be dropped."""
        text = SAMPLE_10K + (
            "\nItem 4: Mine Safety Disclosures\n"
            "We have no mine safety violations to report. " * 12
        )
        sections = select_sections(find_sections(text))
        assert "4" not in [s.item for s in sections]


class TestChunking:
    def test_chunks_respect_size_bound(self):
        section = Section(item="1", title="Business", text="x" * 10_000, start=0, end=10_000)
        chunks = chunk_section(section, "src", max_chars=1000)
        assert all(c.chars <= 1000 for c in chunks)
        assert len(chunks) >= 10

    def test_chunk_ids_and_locators(self):
        section = Section(item="1", title="Business", text="a" * 5000, start=0, end=5000)
        chunks = chunk_section(section, "sec_10k", max_chars=1000)
        assert chunks[0].chunk_id.startswith("sec_10k#i1p1")
        assert "part 1/" in chunks[0].locator

    def test_oversized_paragraph_is_split(self):
        section = Section(item="1", title="Business", text="y" * 9000, start=0, end=9000)
        chunks = chunk_section(section, "src", max_chars=2000)
        assert all(c.chars <= 2000 for c in chunks)

    def test_dedupe_removes_exact_duplicates(self):
        """Byte-identical bodies collapse to one, keeping the first."""
        body = (
            "We operate in two reportable segments and our deposition and etch "
            "products serve leading edge logic and memory customers worldwide. "
        ) * 3
        a = Chunk("a", body, "1", "Business", "s", 1, 2, len(body))
        b = Chunk("b", body, "1", "Business", "s", 2, 2, len(body))
        kept, dropped = dedupe_chunks([a, b])
        assert len(kept) == 1
        assert kept[0].chunk_id == "a"
        assert dropped == 1

    def test_dedupe_removes_near_duplicates(self):
        """Lightly edited repeats must also collapse - exact hashing misses them."""
        base = (
            "We operate in two reportable segments and our deposition and etch "
            "products serve leading edge logic and memory customers worldwide. "
        )
        a = Chunk("a", base * 3, "1", "Business", "s", 1, 1, len(base) * 3)
        b = Chunk("b", base * 3 + "Minor trailing addition.", "1A", "Risk", "s", 1, 1, 0)
        kept, dropped = dedupe_chunks([a, b])
        assert len(kept) == 1
        assert dropped == 1

    def test_partial_overlap_is_not_treated_as_duplicate(self):
        """Partial overlap must NOT be collapsed.

        Two chunks covering different halves of the same repeated paragraph share
        about half their vocabulary. Treating that as duplication would delete
        genuinely distinct evidence, so the threshold is deliberately high enough
        to keep them.
        """
        first = Chunk(
            "p1",
            "Deposition equipment for leading edge logic customers in Taiwan. " * 3,
            "1",
            "Business",
            "s",
            1,
            1,
            0,
        )
        second = Chunk(
            "p2",
            "Ion implantation services for memory manufacturers in Korea. " * 3,
            "1",
            "Business",
            "s",
            2,
            1,
            0,
        )
        kept, dropped = dedupe_chunks([first, second])
        assert len(kept) == 2
        assert dropped == 0

    def test_identical_chunks_across_sections_dedupe(self):
        text = "We compete with Lam Research in etch." * 20
        a = chunk_section(Section("1", "Business", text, 0, len(text)), "s", 400)
        b = chunk_section(Section("1A", "Risk", text, 0, len(text)), "s", 400)
        kept, dropped = dedupe_chunks(a + b)
        assert dropped > 0

    def test_plan_respects_total_budget(self):
        text = "deposition etch metrology customer revenue. " * 400
        section = Section("1", "Business", text, 0, len(text))
        plan = plan_extraction([section], "s", max_total_chars=5000)
        assert plan.total_chars <= 5000

    def test_plan_respects_section_budget(self):
        """Without per-section caps, risk factors crowd out product detail."""
        biz = Section("1", "Business", "deposition customer revenue. " * 500, 0, 15000)
        risk = Section("1A", "Risk", "risk factor boilerplate text. " * 500, 0, 15000)
        plan = plan_extraction(
            [biz, risk], "s", section_budgets={"1": 4000, "1A": 1000}
        )
        by_section: dict[str, int] = {}
        for c in plan.chunks:
            by_section[c.section_item] = by_section.get(c.section_item, 0) + c.chars
        assert by_section.get("1", 0) <= 4000
        assert by_section.get("1A", 0) <= 1000

    def test_relevance_scoring_prefers_topical_text(self):
        relevant = Section("1", "B", "deposition etch metrology process control", 0, 44)
        irrelevant = Section("1", "B", "the quick brown fox jumps over", 0, 29)
        a = chunk_section(relevant, "s", 400)[0]
        b = chunk_section(irrelevant, "s", 400)[0]
        assert score_chunk(a) > score_chunk(b)


class TestQuoteVerification:
    """The anti-hallucination gate."""

    def test_exact_quote_accepted(self):
        text = "We operate in two reportable segments: Semiconductor Systems."
        ok, _ = _verify_quote(text, "We operate in two reportable segments")
        assert ok

    def test_whitespace_and_case_tolerated(self):
        """Filings break lines arbitrarily, so whitespace must normalise."""
        text = "Net revenue\n\nincreased   in fiscal 2025"
        ok, _ = _verify_quote(text, "net revenue increased in fiscal 2025")
        assert ok

    def test_invented_quote_rejected(self):
        text = "We operate in two reportable segments."
        ok, reason = _verify_quote(text, "We dominate the entire market")
        assert not ok
        assert "not found" in reason

    def test_reworded_quote_rejected(self):
        """Paraphrase is not support. Only verbatim text is."""
        text = "Net revenue declined by 16 percent in China."
        ok, _ = _verify_quote(text, "Revenue in China fell by 16%")
        assert not ok

    def test_short_quote_rejected(self):
        """A two-word 'quote' cannot establish support."""
        ok, reason = _verify_quote("deposition and etch", "deposition")
        assert not ok
        assert "too short" in reason

    def test_missing_quote_rejected(self):
        ok, _ = _verify_quote("some text here that is long enough", None)
        assert not ok

    def test_normalisation_collapses_whitespace(self):
        assert _normalise_quote("a  b\n c") == "a b c"


class TestClaimIds:
    def test_deterministic(self):
        """Identical inputs must yield identical ids so cached runs stay valid."""
        a = _claim_id("src", "chunk", 1, "claim text")
        b = _claim_id("src", "chunk", 1, "claim text")
        assert a == b
        assert a.startswith("c_")

    def test_varies_with_content(self):
        assert _claim_id("s", "c", 1, "a") != _claim_id("s", "c", 2, "a")


class TestJsonRecovery:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        assert extract_json('Sure:\n{"a": 1}\nDone') == {"a": 1}

    def test_array(self):
        assert extract_json("[1, 2]") == [1, 2]

    @pytest.mark.parametrize("bad", ["", "not json", '{"broken":'])
    def test_invalid_rejected(self, bad):
        with pytest.raises(LlmError):
            extract_json(bad)


class TestProcessStepCoercion:
    ALLOWED = ["Deposition", "Etch", "CMP", "Advanced Packaging"]

    def test_exact_match(self):
        assert _coerce_process_steps(["Deposition"], self.ALLOWED) == [
            ProcessStep.DEPOSITION
        ]

    def test_case_insensitive(self):
        assert _coerce_process_steps(["deposition"], self.ALLOWED) == [
            ProcessStep.DEPOSITION
        ]

    def test_invented_step_dropped(self):
        """An invented process step must never reach the report."""
        assert _coerce_process_steps(["Quantum Annealing"], self.ALLOWED) == []

    def test_out_of_scope_step_dropped(self):
        """A real step outside the company's configured scope is not allowed."""
        assert _coerce_process_steps(["Lithography"], self.ALLOWED) == []

    def test_deduplicates(self):
        out = _coerce_process_steps(["Etch", "etch", "ETCH"], self.ALLOWED)
        assert out == [ProcessStep.ETCH]


class TestCitationCoercion:
    def test_dangling_citations_stripped(self):
        items, dropped = _coerce_evidence(
            [{"name": "X", "sources": ["c_real", "c_fake"]}],
            ProductItem,
            {"c_real"},
        )
        assert items[0].sources == ["c_real"]
        assert dropped == 0

    def test_uncited_item_dropped(self):
        """An assertion with no resolvable citation is exactly what we prevent."""
        items, dropped = _coerce_evidence(
            [{"name": "X", "sources": ["c_fake"]}], ProductItem, {"c_real"}
        )
        assert items == []
        assert dropped == 1

    def test_malformed_item_dropped(self):
        items, dropped = _coerce_evidence(
            [{"not_a_product": True}], ProductItem, {"c_real"}
        )
        assert items == []
        assert dropped == 1


class TestLedgerAndValidation:
    def _profile(self, **kwargs) -> CompanyProfile:
        return CompanyProfile(
            company=CompanyIdentity(name="Test Co", ticker="TST"),
            **kwargs,
        )

    def test_ledger_text_includes_ids_and_quotes(self):
        claim = Evidence(
            claim_id="c_abc1234567",
            text="Revenue grew in China.",
            source_id="sec_10_k_2025",
            excerpt="China revenue grew",
            topic="geography",
        )
        text = build_ledger_text([claim])
        assert "c_abc1234567" in text
        assert "geography" in text

    def test_dangling_citation_is_an_error(self):
        profile = self._profile(
            evidence=[
                Evidence(
                    claim_id="c_real000001",
                    text="t",
                    source_id="s",
                )
            ],
            products=[
                ProductItem(name="P", sources=["c_missing000"])
            ],
        )
        report = validate_profile(profile)
        assert not report.ok
        assert any("not in the ledger" in e for e in report.errors)

    def test_disclosed_customer_without_source_is_an_error(self):
        """The specification's explicit rule against invented customers."""
        profile = self._profile(
            customers=[
                CustomerInsight(
                    name="Invented Customer",
                    disclosure=CustomerDisclosure.DISCLOSED,
                    sources=[],
                )
            ]
        )
        report = validate_profile(profile)
        assert not report.ok
        assert any("DISCLOSED" in e for e in report.errors)

    def test_inferred_customer_without_reasoning_is_an_error(self):
        profile = self._profile(
            evidence=[
                Evidence(claim_id="c_x000000001", text="t", source_id="s")
            ],
            customers=[
                CustomerInsight(
                    name="Foundries",
                    disclosure=CustomerDisclosure.INFERRED_CATEGORY,
                    sources=["c_x000000001"],
                    reasoning=None,
                )
            ],
        )
        report = validate_profile(profile)
        assert not report.ok
        assert any("reasoning" in e for e in report.errors)

    def test_valid_profile_passes(self):
        profile = self._profile(
            evidence=[Evidence(claim_id="c_ok00000001", text="t", source_id="s")],
            products=[
                ProductItem(
                    name="P",
                    process_steps=[ProcessStep.DEPOSITION],
                    sources=["c_ok00000001"],
                )
            ],
            customers=[
                CustomerInsight(
                    name="Foundries",
                    disclosure=CustomerDisclosure.INFERRED_CATEGORY,
                    sources=["c_ok00000001"],
                    reasoning="Inferred from disclosed end markets.",
                )
            ],
        )
        report = validate_profile(profile)
        assert report.ok, report.errors

    def test_duplicate_claim_ids_flagged(self):
        profile = self._profile(
            evidence=[
                Evidence(claim_id="c_dup0000001", text="a", source_id="s"),
                Evidence(claim_id="c_dup0000001", text="b", source_id="s"),
            ]
        )
        report = validate_profile(profile)
        assert not report.ok
