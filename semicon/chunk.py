"""Section extraction, chunking, deduplication and selective context building.

This module is where token cost is decided. It runs entirely in Python - no LLM
is involved - so the cost of a run is bounded *before* the first API call rather
than discovered afterwards.

The pipeline it serves:

    full filing text (324k chars)
      -> section boundaries        (Item 1, 1A, 7, ...)
      -> keep only relevant items  (~60k chars)
      -> paragraph-aware chunks    (~3k chars each)
      -> dedupe near-identical text
      -> top-N chunks by relevance

Every step is inspectable via ``python main.py <company> --preview``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Section discovery
# --------------------------------------------------------------------------- #

# Real 10-K item headings. AMAT writes "Item 1: Business" where most filers write
# "Item 1. Business", so both separators are accepted. Note that AMAT pads the
# heading with non-breaking spaces ("Item 1:&#160;&#160;&#160;Business"), which
# reach us as ordinary spaces - so the title must tolerate leading whitespace.
_ITEM_RE = re.compile(
    r"^Item\s+(\d{1,2}[A-C]?)\s*[:.]\s+(\S.{0,90})$",
    re.IGNORECASE | re.MULTILINE,
)

# Items worth sending to an LLM, in priority order, with why.
#
# Deliberately excluded:
#   Item 8  - financial statements. XBRL already gives exact numbers; sending
#             these tables would cost tokens to reproduce data we hold precisely.
#   Item 16 - Form 10-K Summary. In practice this swallows everything after the
#             last item heading, including the exhibit index.
#   Items 10-14 - governance and compensation. Not part of the framework.
RELEVANT_SECTIONS: dict[str, tuple[str, str]] = {
    "1": (
        "Business",
        "Products, technologies, process steps, customers, competitors, strategy.",
    ),
    "7": (
        "MD&A",
        "Management commentary on demand drivers, segment performance, outlook.",
    ),
    "7A": ("Market Risk", "Exposure to cycles, rates, and demand volatility."),
    "2": ("Properties", "Manufacturing and geographic footprint."),
    "3": ("Legal Proceedings", "Material litigation affecting the business."),
    "1A": ("Risk Factors", "Business risks and competitive threats."),
}

# Section-level character budgets for extraction. Business and MD&A carry nearly
# all of the framework's signal; risk factors are long, repetitive and mostly
# generic, so they get a smaller share. Without per-section caps, Item 1A alone
# (69k chars) would crowd out the product detail the report exists to explain.
SECTION_BUDGET_CHARS: dict[str, int] = {
    "1": 13_000,
    "7": 10_000,
    "7A": 2_000,
    "2": 1_500,
    "3": 1_500,
    "1A": 6_000,
}


@dataclass
class Section:
    """A named region of a filing."""

    item: str
    title: str
    text: str
    start: int
    end: int

    @property
    def key(self) -> str:
        return f"Item {self.item}"

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Chunk:
    """An extraction-sized unit of text with full provenance."""

    chunk_id: str
    text: str
    section_item: str
    section_title: str
    source_id: str
    part: int
    of_parts: int
    source_chars: int
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[
                :16
            ]

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def est_tokens(self) -> int:
        """Rough token estimate: ~4 characters per token for English prose.

        Used for budgeting and previews only. It is intentionally a slight
        over-estimate so the budget guard errs toward caution.
        """
        return self.chars // 4

    @property
    def locator(self) -> str:
        base = f"{self.section_title} (Item {self.section_item})"
        return f"{base}, part {self.part}/{self.of_parts}" if self.of_parts > 1 else base


# Not every filer keeps its Item headings. KLA's Form 10-K lists every item in the
# table of contents but then runs the body under bare "PART I" / "PART II" markers
# with no repeated headings, so an Item-only section scan finds nothing at all and
# the filing yields zero evidence. PART markers are universal across 10-Ks, so
# they are used as a fallback structural boundary.
_PART_RE = re.compile(r"^PART\s+(I{1,3}V?|IV)\s*$", re.IGNORECASE | re.MULTILINE)


def find_part_sections(text: str, min_chars: int = 400) -> list[Section]:
    """Fallback sectioning on PART boundaries.

    Every 10-K is organised as Parts I-IV even when item headings are absent. The
    table-of-contents copy of a PART marker is skipped by the same length test used
    for items: a contents entry has no body text after it.
    """
    matches = list(_PART_RE.finditer(text))
    if not matches:
        return []

    sections: list[Section] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end]
        if len(body) < min_chars or body.count(". ") < 3:
            continue
        sections.append(
            Section(
                item=f"PART {m.group(1).upper()}",
                title=f"Part {m.group(1).upper()}",
                text=body,
                start=m.start(),
                end=end,
            )
        )
    return sections


def find_sections(text: str) -> list[Section]:
    """Locate the annual report's Item sections.

    Every Item heading appears twice: once in the table of contents and once in
    the body. The body occurrence is identified by having substantial text after
    it, which is what distinguishes it from a contents entry.

    Note: do NOT try to truncate the document at a "Signatures" or "Exhibit Index"
    marker. Those strings also appear in the table of contents, which sits near
    the very start, so truncating there discards the entire document body. The
    length filter below is what removes contents entries, and it is sufficient.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return []

    limit = len(text)
    sections: list[Section] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else limit
        body = text[start:end]

        # A table-of-contents entry is short and has no sentence content.
        if len(body) < 400 or body.count(". ") < 3:
            continue
        sections.append(
            Section(
                item=m.group(1).upper(),
                title=m.group(2).strip(),
                text=body,
                start=start,
                end=end,
            )
        )
    return sections


def resolve_sections(text: str) -> list[Section]:
    """Find usable sections in a filing, whatever structure it uses.

    Three tiers, because filings differ by more than they look:

    1. **Item headings** (10-K). Items 1, 1A, 7 and so on, with a priority order.
    2. **PART boundaries.** Some 10-Ks list items only in the table of contents and
       run the body under bare "PART I"/"PART II" markers, so an Item-only scan
       finds nothing at all.
    3. **The whole document** as one section, for foreign private issuers. A Form
       20-F has no Item structure whatsoever - ASML's is 1.35 MB with zero Item
       headings - and without this tier such a filing yields no text and therefore
       no claims, which looked like an extraction failure but was a discovery gap.

    Peer filings already used tiers 2-3 for the same reason; the subject path did
    not, which is why ASML produced an empty report while its peers parsed.
    """
    items = select_sections(find_sections(text))
    if items:
        return items

    parts = find_part_sections(text)
    if parts:
        return parts

    if len(text) < 500:
        return []
    return [
        Section(
            item="DOC",
            title="Filing document",
            text=text,
            start=0,
            end=len(text),
        )
    ]


def select_sections(
    sections: list[Section], wanted: dict[str, tuple[str, str]] | None = None
) -> list[Section]:
    """Keep only the sections relevant to the framework, newest body copy first.

    When an item appears more than once (which happens when a heading recurs in a
    later cross-reference), the longest instance wins - that is the real section
    rather than a passing mention.
    """
    wanted = wanted or RELEVANT_SECTIONS
    best: dict[str, Section] = {}
    for s in sections:
        if s.item not in wanted:
            continue
        if s.item not in best or s.chars > best[s.item].chars:
            best[s.item] = s
    # Preserve the priority order declared in RELEVANT_SECTIONS.
    order = list(wanted.keys())
    return sorted(best.values(), key=lambda s: order.index(s.item))


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_section(
    section: Section, source_id: str, max_chars: int = 3200
) -> list[Chunk]:
    """Split a section into paragraph-aligned chunks.

    Splitting on paragraph boundaries rather than fixed offsets keeps each chunk
    semantically coherent, which measurably improves extraction quality: a chunk
    that begins mid-sentence forces the model to guess at context.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(section.text) if p.strip()]
    if not paragraphs:
        return []

    groups: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # A single oversized paragraph becomes its own chunk, split if needed.
        if len(para) > max_chars:
            if current:
                groups.append("\n\n".join(current))
                current, current_len = [], 0
            for i in range(0, len(para), max_chars):
                groups.append(para[i : i + max_chars])
            continue
        if current_len + len(para) > max_chars and current:
            groups.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        groups.append("\n\n".join(current))

    total = len(groups)
    return [
        Chunk(
            chunk_id=f"{source_id}#i{section.item}p{i + 1}",
            text=g,
            section_item=section.item,
            section_title=section.title,
            source_id=source_id,
            part=i + 1,
            of_parts=total,
            source_chars=section.chars,
        )
        for i, g in enumerate(groups)
    ]


_NORMALISE_WS = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalised_bag(text: str) -> frozenset[str]:
    """Word set for similarity comparison."""
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dedupe_chunks(
    chunks: list[Chunk],
    *,
    near_duplicate_threshold: float = 0.82,
    min_words_for_near_check: int = 40,
) -> tuple[list[Chunk], int]:
    """Drop chunks that duplicate content already seen.

    Two passes, because filings repeat themselves in two different ways:

    1. **Exact.** Risk factors restate business descriptions word for word across
       different sections. A content-hash match catches these for free.
    2. **Near.** The same discussion often reappears lightly edited, for example
       an MD&A summary of a segment table. Jaccard similarity over word sets
       catches these, which exact hashing cannot.

    Similarity is a word-set comparison rather than an embedding: it is free,
    deterministic, and tuned by the threshold below. Sending near-duplicates
    costs tokens and, worse, inflates the apparent number of independent sources
    supporting a claim.
    """
    seen_hashes: set[str] = set()
    kept: list[Chunk] = []
    kept_bags: list[tuple[frozenset[str], int]] = []
    dropped = 0

    for c in chunks:
        if c.content_hash in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(c.content_hash)

        words = _WORD_RE.findall(c.text.lower())
        # Short chunks are compared on exact hash only; their word sets are too
        # small for Jaccard to be meaningful.
        if len(words) >= min_words_for_near_check:
            bag = frozenset(words)
            duplicate = False
            for other_bag, other_words in kept_bags:
                # Cheap guard before the set operation, comparing raw word counts
                # (not set sizes, which collapse repeated words and would make a
                # 60-word chunk with 18 unique words look comparable to a 21-word
                # chunk). A truncated repeat is still a duplicate, so this sits at
                # 0.4 rather than something tighter.
                if min(len(words), other_words) / max(len(words), other_words, 1) < 0.4:
                    continue
                if _jaccard(bag, other_bag) >= near_duplicate_threshold:
                    duplicate = True
                    break
            if duplicate:
                dropped += 1
                continue
            kept_bags.append((bag, len(words)))
        kept.append(c)

    return kept, dropped


# Terms that indicate a chunk is relevant to the framework's questions.
_RELEVANCE_TERMS: dict[str, int] = {
    "deposition": 3, "etch": 3, "lithograph": 3, "metrology": 3, "inspection": 3,
    "implant": 2, "chemical mechanical": 2, "cmp": 2, "packaging": 3,
    "process control": 3, "wafer": 2, "node": 2, "transistor": 2,
    "segment": 2, "net revenue": 2, "net sales": 2, "revenue": 1,
    "customer": 3, "foundry": 3, "memory": 2, "dram": 2, "nand": 2, "hbm": 3,
    "competitor": 3, "compete": 2, "competition": 2, "market share": 2,
    "ai": 2, "artificial intelligence": 3, "advanced packaging": 3,
    "backlog": 2, "bookings": 2, "research and development": 2, "product": 1,
    "technology": 2, "roadmap": 2, "manufactur": 1, "supply": 1,
    "acquisition": 1, "divest": 1, "strategy": 2, "growth": 1,
}


def score_chunk(chunk: Chunk, terms: dict[str, int] | None = None) -> int:
    """Lexical relevance score for a chunk.

    Deliberately lexical rather than embedding-based: for a single 10-K the
    document is small, the vocabulary is domain-specific and stable, and a
    keyword score is free, deterministic, debuggable and good enough. An
    embedding index would add cost and a dependency to solve a problem we do not
    measurably have.

    The score is a raw keyword tally with no length normalisation. An earlier
    version divided by chunk length, which made a 55-character page header
    ("Lam Research Corporation 2026 10-K 7  Table of Contents") outrank a
    3,200-character product discussion: with only a couple of term hits, dividing
    by a tiny length produced a huge score. Raw counts plus a minimum-size filter
    (see ``min_chunk_chars``) is both simpler and correct.
    """
    terms = terms or _RELEVANCE_TERMS
    low = chunk.text.lower()
    return sum(weight * low.count(term) for term, weight in terms.items())


# Chunks smaller than this are filing page furniture - running headers, page
# numbers, "Table of Contents" lines - and are never worth sending to a model.
# Applied Materials' 10-K produced a 17-character chunk reading only
# "Table of Contents" that consumed a slot in the extraction budget.
MIN_CHUNK_CHARS = 200


@dataclass
class SelectionPlan:
    """The result of planning which text will be sent to the model."""

    chunks: list[Chunk] = field(default_factory=list)
    dropped_duplicates: int = 0
    dropped_fragments: int = 0
    total_chunks_available: int = 0
    considered_sections: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(c.chars for c in self.chunks)

    @property
    def est_tokens(self) -> int:
        return sum(c.est_tokens for c in self.chunks)


def plan_extraction(
    sections: list[Section],
    source_id: str,
    *,
    max_chars_per_chunk: int = 3200,
    max_chunks: int | None = None,
    max_total_chars: int | None = None,
    section_budgets: dict[str, int] | None = None,
) -> SelectionPlan:
    """Build the exact set of chunks that extraction will send.

    Parameters are hard ceilings, not targets: the plan stops at whichever limit
    binds first, so a pathological filing cannot produce an unbounded prompt.

    Budgets are applied per section *and* in total. The per-section caps matter
    because relevance scoring can otherwise let one long repetitive section (risk
    factors) consume the entire global budget, crowding out product detail.
    """
    plan = SelectionPlan(considered_sections=[s.key for s in sections])
    budgets = section_budgets if section_budgets is not None else SECTION_BUDGET_CHARS

    all_chunks: list[Chunk] = []
    for section in sections:
        all_chunks.extend(
            chunk_section(section, source_id, max_chars=max_chars_per_chunk)
        )

    plan.total_chunks_available = len(all_chunks)
    all_chunks, dropped = dedupe_chunks(all_chunks)
    plan.dropped_duplicates = dropped

    # Drop page furniture before ranking, so budget slots go to prose. If a
    # document is entirely short fragments (unusual), keep them rather than
    # returning nothing.
    substantial = [c for c in all_chunks if c.chars >= MIN_CHUNK_CHARS]
    if substantial:
        plan.dropped_fragments = len(all_chunks) - len(substantial)
        all_chunks = substantial

    priority = {s.item: i for i, s in enumerate(sections)}
    # Within a section, most relevant first; then restore document order after.
    all_chunks.sort(key=lambda c: (priority.get(c.section_item, 99), -score_chunk(c)))

    chosen: list[Chunk] = []
    running_total = 0
    running_by_section: dict[str, int] = {}

    for c in all_chunks:
        if max_chunks is not None and len(chosen) >= max_chunks:
            break
        section_cap = budgets.get(c.section_item)
        section_used = running_by_section.get(c.section_item, 0)
        if section_cap is not None and section_used + c.chars > section_cap:
            continue
        if max_total_chars is not None and running_total + c.chars > max_total_chars:
            continue
        chosen.append(c)
        running_total += c.chars
        running_by_section[c.section_item] = section_used + c.chars

    # Re-sort the chosen chunks back into document order so the model reads the
    # business description before the risk factors, as a human would.
    chosen.sort(key=lambda c: (priority.get(c.section_item, 99), c.part))
    plan.chunks = chosen
    return plan


def build_evidence_context(
    sections: list[Section], *, max_chars: int = 40_000
) -> str:
    """Compact raw text for the synthesis prompt.

    Synthesis reasons over extracted claims, but it also benefits from the
    filing's own prose for the MD&A and risk sections, which are short. This
    trims those sections to a budget rather than sending them whole.
    """
    parts: list[str] = []
    used = 0
    for s in sections:
        if s.item not in {"7", "7A", "1A"}:
            continue
        remaining = max_chars - used
        if remaining <= 500:
            break
        body = s.text[:remaining]
        parts.append(f"--- {s.key}: {s.title} ---\n{body}")
        used += len(body)
    return "\n\n".join(parts)
