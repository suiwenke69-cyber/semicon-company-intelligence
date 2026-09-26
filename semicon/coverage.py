"""Coverage tracking: which analytical categories the evidence actually supports.

Why this module exists
----------------------
A single extraction pass produced 66 claims for Applied Materials, of which 15
were market drivers and only 2 were products. For a report whose stated purpose is
"Technology -> Product -> Market -> Customer -> Competition -> Business", that is a
badly unbalanced evidence base: the pipeline looked like a news summariser rather
than a product-intelligence tool.

The naive fix - instruct the model to "produce 10 products" - is dangerous, because
it creates pressure to invent. This module takes the alternative approach:

* Each category has a **coverage target**, an expectation about how much evidence a
  well-documented semiconductor company should yield.
* Targets are **advisory, never enforced**. Extraction is still gated on a
  verifiable quote, so a category can legitimately come in short.
* Gaps are measured and **fed back as a hint to the next extraction pass**, which
  is how the pipeline recovers missing categories without lowering the evidence bar.
* Shortfalls are **reported to the reader** rather than hidden, so an
  under-supported section is visible as a limitation instead of looking like an
  absence of facts.

An unmet target means "the sources did not support this", not "the model failed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Evidence, EXTRACTION_TOPICS

# Categories that matter for a B2B commercial-intelligence report, mapped onto the
# extraction topics that feed them.
#
# Two topics are deliberately NOT coverage-tracked:
#   company_overview and financial_commentary are ambient - they appear in almost
#   every chunk, so a target would be trivially met and would measure nothing.
CATEGORY_TOPICS: dict[str, tuple[str, ...]] = {
    "products": ("product",),
    "process_steps": ("process_step",),
    "technologies": ("technology",),
    "customer_needs": ("customer",),
    "market_drivers": ("market_driver",),
    "competitors": ("competitor",),
    "commercial_implications": ("strategy", "financial_commentary"),
}

# Minimum supported claims a well-documented semiconductor company should yield per
# category. Set from what the sources actually contain for a large-cap equipment
# vendor, not from what would look impressive:
#
#   products    - a 10-K names product categories, an earnings release names
#                 platforms; 3 distinct supported product claims is a realistic floor.
#   competitors - annual reports frequently name NO competitor at all (AMAT names
#                 none), so this target is intentionally the lowest.
#   customer    - requires either a named customer or a described category.
TARGETS: dict[str, int] = {
    "products": 3,
    "process_steps": 2,
    "technologies": 2,
    "customer_needs": 2,
    "market_drivers": 2,
    "competitors": 1,
    "commercial_implications": 2,
}

# Below this, a category is considered absent rather than merely thin, and is worth
# spending a targeted pass on.
SEVERE_THRESHOLD = 1


@dataclass
class CategoryCoverage:
    """Coverage for one analytical category."""

    category: str
    count: int
    target: int
    topics: tuple[str, ...]

    @property
    def met(self) -> bool:
        return self.count >= self.target

    @property
    def absent(self) -> bool:
        return self.count < SEVERE_THRESHOLD

    @property
    def shortfall(self) -> int:
        return max(0, self.target - self.count)

    def describe(self) -> str:
        state = "met" if self.met else ("absent" if self.absent else "thin")
        return f"{self.category}: {self.count}/{self.target} ({state})"


@dataclass
class CoverageReport:
    """Coverage across all tracked categories, plus the topic-level histogram."""

    categories: dict[str, CategoryCoverage] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    total_claims: int = 0

    @property
    def met_count(self) -> int:
        return sum(1 for c in self.categories.values() if c.met)

    @property
    def unmet(self) -> list[CategoryCoverage]:
        return [c for c in self.categories.values() if not c.met]

    @property
    def absent(self) -> list[CategoryCoverage]:
        return [c for c in self.categories.values() if c.absent]

    @property
    def is_balanced(self) -> bool:
        return not self.unmet

    def summary(self) -> str:
        return (
            f"{self.total_claims} claims; "
            f"{self.met_count}/{len(self.categories)} categories at target"
        )

    def gaps_for_hint(self) -> list[str]:
        """Categories worth a targeted follow-up pass, worst first.

        Only genuinely thin or absent categories are returned. Asking for a
        category that is already at target would waste tokens and invite padding.
        """
        return [
            c.category for c in sorted(self.unmet, key=lambda c: c.count)
        ]

    def notes(self) -> list[str]:
        """Reader-facing statements about unmet targets."""
        out = []
        for c in self.unmet:
            out.append(
                f"evidence shortfall: only {c.count} supported {c.category.replace('_', ' ')} "
                f"claim(s) found against a target of {c.target}; the sources reviewed "
                f"may not disclose this area"
            )
        return out


def _topic_of(claim: Evidence) -> str:
    return (claim.topic or "company_overview").strip().lower()


def measure_coverage(claims: list[Evidence]) -> CoverageReport:
    """Compute coverage for a claim ledger."""
    topic_counts: dict[str, int] = {}
    for c in claims:
        topic_counts[_topic_of(c)] = topic_counts.get(_topic_of(c), 0) + 1

    categories: dict[str, CategoryCoverage] = {}
    for category, topics in CATEGORY_TOPICS.items():
        count = sum(topic_counts.get(t, 0) for t in topics)
        categories[category] = CategoryCoverage(
            category=category,
            count=count,
            target=TARGETS.get(category, 1),
            topics=topics,
        )

    return CoverageReport(
        categories=categories,
        topic_counts=topic_counts,
        total_claims=len(claims),
    )


def coverage_hint(report: CoverageReport, process_steps: list[str] | None = None) -> str:
    """Build the per-chunk instruction that steers extraction toward gaps.

    This is the mechanism that improves balance WITHOUT lowering the evidence bar.
    It tells the model what is currently missing and asks it to look specifically
    for that - while the quote gate independently guarantees that nothing gets
    invented to satisfy the request.

    Once every category is at target, the hint collapses to a general instruction,
    so later chunks are not nudged toward padding.
    """
    gaps = report.gaps_for_hint()
    steps = ", ".join(process_steps) if process_steps else "the standard process steps"

    if not gaps:
        return (
            "Extract any well-supported claims in this passage. Prefer ones that "
            "are specific and quantitative."
        )

    wants = {
        "products": (
            "specific products, product families or platforms, and what process "
            "step each addresses"
        ),
        "process_steps": f"named process steps, chosen only from: {steps}",
        "technologies": "named technologies or technical capabilities and what makes them different",
        "customer_needs": (
            "customer categories, what customers require, and what problems they "
            "need solved"
        ),
        "market_drivers": "demand drivers such as AI infrastructure, HBM, advanced packaging, leading-edge logic, or memory cycles",
        "competitors": "named competitors, or explicit statements about where competition comes from",
        "commercial_implications": "management commentary on strategy, growth, pricing, backlog or demand outlook",
    }

    lines = [
        wants[g] for g in gaps if g in wants
    ]
    if not lines:
        return "Extract any well-supported claims in this passage."

    missing = "\n".join(f"  - {line}" for line in lines)
    return (
        "This passage may contain evidence that earlier passes did not surface. "
        "Look specifically for:\n"
        f"{missing}\n"
        "Only include items you can support with a verbatim quote. If this passage "
        "contains no such evidence, return an empty list rather than stretching it."
    )
