"""Regression tests for defects found only by running against the live API.

Every test here corresponds to something the 98-test offline suite did NOT catch
and that surfaced during the first real end-to-end runs. That is the point: unit
tests over synthetic input missed integration behaviour, and these lock it down.
"""

from __future__ import annotations

import pytest

from semicon.extract import SYSTEM_PROMPT, extract_chunk
from semicon.llm.client import LlmClient
from semicon.models import (
    BusinessSegment,
    CompanyIdentity,
    CompanyProfile,
    Evidence,
    GeographicRevenue,
    Source,
    SourceKind,
)
from semicon.pipeline import _merge_analysis_segments
from semicon.validate import validate_profile


class TestSegmentMergePreservesDeterministicRevenue:
    """Defect 1: the model's qualitative segments REPLACED the parsed table.

    The pipeline computed $20,798M for Semiconductor Systems, then overwrote the
    segment with the model's narrative object, which has no revenue field. The
    report rendered the segment table as entirely "n/a" while the correct numbers
    sat unused a layer below.
    """

    def test_revenue_survives_merge(self):
        deterministic = [
            BusinessSegment(
                name="Semiconductor Systems",
                revenue_millions=20798.0,
                revenue_pct_of_total=73.0,
                revenue_change_pct=4.5,
                revenue_note="Revenue as reported in FY2025",
                source_ids=["sec_10_k_2025"],
            )
        ]
        from_model = [
            BusinessSegment(
                name="Semiconductor Systems",
                description="Deposition, etch and CMP equipment.",
                products=["Producer"],
                sources=["c_abc1234567"],
            )
        ]
        merged = _merge_analysis_segments(deterministic, from_model)
        assert len(merged) == 1
        seg = merged[0]
        # Deterministic numbers preserved ...
        assert seg.revenue_millions == 20798.0
        assert seg.revenue_pct_of_total == 73.0
        assert seg.revenue_note == "Revenue as reported in FY2025"
        # ... and the model's narrative merged in.
        assert seg.description == "Deposition, etch and CMP equipment."
        assert seg.products == ["Producer"]

    def test_citations_from_both_sources_unioned(self):
        deterministic = [
            BusinessSegment(name="AGS", revenue_millions=6385.0, source_ids=["sec_10_k_2025"])
        ]
        from_model = [
            BusinessSegment(name="AGS", description="Services.", sources=["c_abc1234567"])
        ]
        merged = _merge_analysis_segments(deterministic, from_model)
        assert merged[0].sources == ["c_abc1234567"]
        assert merged[0].source_ids == ["sec_10_k_2025"]

    def test_model_only_segment_kept_without_invented_revenue(self):
        """Display was reported in FY2024 only; it must not gain figures."""
        merged = _merge_analysis_segments(
            [],
            [BusinessSegment(name="Display", description="Display equipment.", sources=["c_x"])],
        )
        assert merged[0].revenue_millions is None
        assert "No segment revenue row matched" in (merged[0].revenue_note or "")

    def test_name_matching_is_normalised(self):
        deterministic = [BusinessSegment(name="Applied Global Services", revenue_millions=6385.0)]
        from_model = [BusinessSegment(name="applied  global services", description="Services.")]
        merged = _merge_analysis_segments(deterministic, from_model)
        assert len(merged) == 1
        assert merged[0].revenue_millions == 6385.0

    def test_does_not_mutate_the_input_list(self):
        original = BusinessSegment(name="X", revenue_millions=1.0)
        _merge_analysis_segments([original], [BusinessSegment(name="X", description="d")])
        assert original.description is None


class TestProvenanceKindsAreNotConflated:
    """Defect 2: a filing source id was validated as if it were a claim id.

    Deterministic table parses cite the FILING ('sec_10_k_2025'); LLM-derived items
    cite CLAIMS ('c_...'). Storing both in one field made the validator report a
    correct table parse as a dangling citation.
    """

    def _profile(self, **kwargs) -> CompanyProfile:
        return CompanyProfile(
            company=CompanyIdentity(name="Test Co"),
            sources=[
                Source(
                    source_id="sec_10_k_2025",
                    kind=SourceKind.SEC_10K,
                    title="10-K",
                    url="https://sec.gov/x",
                )
            ],
            **kwargs,
        )

    def test_deterministic_segment_source_id_is_valid(self):
        profile = self._profile(
            business_segments=[
                BusinessSegment(
                    name="Semiconductor Systems",
                    revenue_millions=20798.0,
                    source_ids=["sec_10_k_2025"],
                )
            ]
        )
        report = validate_profile(profile)
        assert report.ok, report.errors

    def test_unknown_source_id_is_an_error(self):
        profile = self._profile(
            business_segments=[
                BusinessSegment(name="X", source_ids=["sec_10_k_1900"])
            ]
        )
        report = validate_profile(profile)
        assert not report.ok
        assert any("unknown source id" in e for e in report.errors)

    def test_geographic_source_id_validated(self):
        profile = self._profile(
            geographic_revenue=[
                GeographicRevenue(region="China", revenue_millions=8529.0, source_id="sec_10_k_2025")
            ]
        )
        assert validate_profile(profile).ok

        bad = self._profile(
            geographic_revenue=[
                GeographicRevenue(region="China", revenue_millions=8529.0, source_id="nope")
            ]
        )
        assert not validate_profile(bad).ok

    def test_claim_citation_still_validated_separately(self):
        """A claim id in `sources` must still resolve to the ledger."""
        profile = self._profile(
            evidence=[Evidence(claim_id="c_real000001", text="t", source_id="sec_10_k_2025")],
            business_segments=[
                BusinessSegment(name="X", sources=["c_missing000"])
            ],
        )
        report = validate_profile(profile)
        assert not report.ok
        assert any("not in the ledger" in e for e in report.errors)


class TestExtractionPromptContract:
    """Defect 3: the prompt suppressed risks and competitors entirely.

    The original wording said to ignore "risk-factor genericities", and the model
    over-applied that to every risk. On a real AMAT 10-K whose Risk Factors
    section plainly names customer concentration, geographic concentration and
    export controls, extraction returned ZERO risk claims and ZERO competitor
    claims. These assertions guard the wording that fixed it.
    """

    def test_prompt_affirmatively_requests_risks(self):
        assert "COMPANY-SPECIFIC RISKS" in SYSTEM_PROMPT
        assert "primary source, not boilerplate" in SYSTEM_PROMPT

    def test_prompt_names_concrete_risk_examples(self):
        low = SYSTEM_PROMPT.lower()
        for example in ("customer concentration", "export controls", "supply-chain"):
            assert example in low

    def test_prompt_requests_competitors(self):
        assert "Named competitors" in SYSTEM_PROMPT

    def test_harmful_blanket_instruction_removed(self):
        """The exact phrase that caused the model to drop all risks."""
        assert "risk-factor genericities" not in SYSTEM_PROMPT

    def test_topics_still_closed_and_complete(self):
        from semicon.models import EXTRACTION_TOPICS

        for topic in EXTRACTION_TOPICS:
            assert topic in SYSTEM_PROMPT
        assert "risk" in SYSTEM_PROMPT
        assert "competitor" in SYSTEM_PROMPT


class FakeLlm(LlmClient):
    """Returns canned claims; records nothing and calls no API."""

    def __init__(self, settings, payload):
        super().__init__(settings, api_key="stub")
        self._payload = payload

    def complete_json(self, *, model, system, user, max_tokens=None, use_cache=True):
        return self._payload


class TestQuoteGateOnRealisticOutput:
    """The gate must accept genuine quotes and reject invented ones."""

    def _chunk(self):
        from semicon.chunk import Chunk

        return Chunk(
            chunk_id="s#i1p1",
            text=(
                "A relatively limited number of customers account for a substantial "
                "portion of our business. Our customer base is geographically "
                "concentrated, particularly in China, Taiwan and Korea."
            ),
            section_item="1A",
            section_title="Risk Factors",
            source_id="sec_10_k_2025",
            part=1,
            of_parts=1,
            source_chars=180,
        )

    def test_genuine_quote_survives(self):
        from semicon.config import load_settings

        settings = load_settings()
        client = FakeLlm(
            settings,
            {
                "claims": [
                    {
                        "claim": "A limited number of customers account for a substantial portion of revenue.",
                        "quote": "A relatively limited number of customers account for a substantial portion of our business",
                        "topic": "risk",
                        "confidence": "high",
                    }
                ]
            },
        )
        result = extract_chunk(
            self._chunk(), client=client, settings=settings, company_name="AMAT"
        )
        assert len(result.claims) == 1
        assert result.claims[0].topic == "risk"

    def test_invented_quote_rejected(self):
        from semicon.config import load_settings

        settings = load_settings()
        client = FakeLlm(
            settings,
            {
                "claims": [
                    {
                        "claim": "The company dominates the global market.",
                        "quote": "We hold 90 percent global market share in deposition",
                        "topic": "competitor",
                        "confidence": "high",
                    }
                ]
            },
        )
        result = extract_chunk(
            self._chunk(), client=client, settings=settings, company_name="AMAT"
        )
        assert result.claims == []
        assert result.dropped_invalid == 1
