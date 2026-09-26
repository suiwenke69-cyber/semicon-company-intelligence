"""Peer-company first-party filings, used for competitive-overlap evidence.

Why this exists
---------------
A company's own annual report is a poor source for competitor analysis, and the
Applied Materials run made that concrete: its Form 10-K names **no competitor at
all**, describing rivals only as "small companies that compete in a single region
... to global, diversified companies". The one peer name that appeared (KLA) was
in an executive biography. Competitor analysis built on self-disclosure alone is
therefore structurally thin, not merely under-extracted.

The fix is a small number of **first-party** peer sources: peer annual reports and
earnings materials, which are free, legally unambiguous, and available through the
same EDGAR adapter already in use. Peer filings describe their own products and
process steps, which is what makes technology overlap checkable rather than
asserted.

What this deliberately does NOT do
----------------------------------
* No paywalled analyst research (Yole, TechInsights). Automating retrieval would
  breach their terms; those remain a manual, cited addition.
* No broad web scraping. SEMI publishes some free material, but the stable,
  automatable, unambiguous core is EDGAR.
* No inference about who competes with whom. Peers are declared in
  ``config/settings.yaml``, because guessing a competitive relationship is exactly
  the unsupported claim this project exists to avoid.

Peer evidence is kept in its own ledger, separate from the subject company's. It
never inflates the subject's claim count, and it is labelled with its origin so a
reader can always tell whose filing a statement came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunk import (
    Chunk,
    Section,
    find_part_sections,
    find_sections,
    plan_extraction,
    select_sections,
)
from .config import CompetitorSourceConfig, Settings
from .http import HttpClient, RetrievalError
from .models import Source, SourceKind
from .sources import sec_edgar

# Sections of a peer filing worth reading for competitive overlap. Competition and
# Business describe what the peer makes; MD&A gives current demand framing.
_PEER_SECTIONS = {"1": ("Business", ""), "7": ("MD&A", "")}


@dataclass
class PeerDocument:
    """A retrieved peer filing plus its citation metadata."""

    peer: CompetitorSourceConfig
    source: Source
    artifact: object | None = None
    overlap_focus: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.artifact is not None and self.source.error is None


def _peer_source_id(peer: CompetitorSourceConfig, fiscal_year: int | None) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in peer.name).strip("_")
    suffix = f"_{fiscal_year}" if fiscal_year else ""
    return f"sec_peer_{slug}{suffix}"


def fetch_peer_documents(
    peers: list[CompetitorSourceConfig],
    *,
    client: HttpClient,
    settings: Settings,
    max_documents: int | None = None,
) -> tuple[list[PeerDocument], list[str]]:
    """Retrieve the most recent annual report for each configured peer.

    Failures are returned as warnings rather than raised: peer evidence enriches
    the report, so losing a peer should degrade the competitive section, not abort
    the run.
    """
    documents: list[PeerDocument] = []
    warnings: list[str] = []
    limit = max_documents if max_documents is not None else settings.sources.max_competitor_documents

    for peer in peers[:limit]:
        if not peer.cik:
            warnings.append(f"peer '{peer.name}' has no CIK configured; skipped")
            continue

        try:
            submissions = sec_edgar.fetch_submissions(client, peer.cik)
            selected = sec_edgar.select_filings(submissions, settings.filings)
            annual = [f for f in selected if f.is_annual]
            if not annual:
                warnings.append(
                    f"peer '{peer.name}': no annual report found on EDGAR; skipped"
                )
                continue
            filing = annual[0]
        except RetrievalError as exc:
            warnings.append(f"peer '{peer.name}': {exc}")
            continue

        source = Source(
            source_id=_peer_source_id(peer, filing.fiscal_year),
            kind=SourceKind.SEC_10K,
            title=f"{peer.name} {filing.form} (peer filing, competitive overlap)",
            url=filing.primary_document_url or filing.index_url,
            published=filing.filing_date,
            accession_number=filing.accession_number,
            fiscal_year=filing.fiscal_year,
        )
        try:
            artifact = sec_edgar.fetch_filing_document(client, filing, settings)
        except RetrievalError as exc:
            source.error = str(exc)
            documents.append(PeerDocument(peer=peer, source=source))
            warnings.append(f"peer '{peer.name}': {exc}")
            continue

        source.content_hash = artifact.content_hash
        source.cache_path = str(
            artifact.cache_path.relative_to(settings.cache_root.parent)
        )
        source.retrieved_at = artifact.retrieved_at
        documents.append(
            PeerDocument(
                peer=peer,
                source=source,
                artifact=artifact,
                overlap_focus=list(peer.overlap_focus),
            )
        )

    return documents, warnings


def plan_peer_chunks(
    documents: list[PeerDocument],
    *,
    settings: Settings,
    max_total_chars: int | None = None,
) -> list[Chunk]:
    """Select the peer text worth sending, within a separate budget.

    Peer text has its own character ceiling so that, however many peers are
    configured, competitor context can never crowd out the subject company's own
    evidence. Extracting competitive overlap needs far less text than a full
    company profile does.
    """
    budget = (
        max_total_chars
        if max_total_chars is not None
        else settings.sources.competitor_char_budget
    )
    per_peer = max(2_000, budget // max(1, len(documents)))
    chunks: list[Chunk] = []

    for doc in documents:
        if not doc.ok:
            continue
        text = sec_edgar.html_to_text(doc.artifact.text)  # type: ignore[union-attr]
        sections = [
            s
            for s in select_sections(find_sections(text))
            if s.item in _PEER_SECTIONS
        ]
        budgets = {"1": per_peer, "7": per_peer, "DOC": per_peer}
        if not sections:
            # Not every filer repeats its Item headings in the body. KLA's 10-K
            # runs the body under bare PART markers, so fall back to PART
            # boundaries before falling back to the whole document.
            part_sections = find_part_sections(text)
            if part_sections:
                sections = part_sections
                budgets = {s.item: per_peer for s in part_sections}
        if not sections:
            # Last resort: treat the document as one section so an unfamiliar
            # format degrades quality rather than losing the peer entirely.
            sections = [
                Section(
                    item="DOC",
                    title=f"{doc.peer.name} filing",
                    text=text,
                    start=0,
                    end=len(text),
                )
            ]
        plan = plan_extraction(
            sections,
            doc.source.source_id,
            max_total_chars=per_peer,
            section_budgets=budgets,
        )
        chunks.extend(plan.chunks)

    return chunks


def peer_overlap_brief(
    submissions_by_peer: dict[str, list[str]],
) -> str:
    """Render the declared overlap focus for the synthesis prompt.

    Passing this explicitly keeps competitor reasoning anchored to the areas a
    human reviewer considered genuinely overlapping, rather than letting the model
    characterise the whole peer company.
    """
    lines = []
    for name, focus in submissions_by_peer.items():
        if focus:
            lines.append(f"- {name}: overlapping areas to assess - {', '.join(focus)}")
        else:
            lines.append(f"- {name}: no specific overlap declared")
    return "\n".join(lines)
