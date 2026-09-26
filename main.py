#!/usr/bin/env python3
"""CLI entry point.

    python main.py "Applied Materials"

Steps 1-2 (deterministic retrieval + XBRL financials) run by default and spend
zero tokens. LLM stages are gated behind an explicit flag and are not yet
implemented, so it is impossible to spend money by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semicon import pipeline
from semicon.config import load_settings, project_root
from semicon.formatting import format_money
from semicon.http import RetrievalError
from semicon.llm.client import BudgetExceeded, LlmClient, LlmError
from semicon.models import CompanyProfile
from semicon.render import render_report, write_report as render_write
from semicon.resolve import ResolutionError


def _fmt_money(value: float | None, unit: str) -> str:
    """Delegates to the shared formatter so CLI and report cannot diverge."""
    return format_money(value, unit)


def print_summary(
    profile: CompanyProfile, artifacts: pipeline.RunArtifacts, quarterly: bool = False
) -> None:
    """Console summary. Deliberately dense - this is the developer-facing view."""
    c = profile.company
    line = "=" * 72

    print(f"\n{line}\n  {c.name}  ({c.ticker or 'n/a'})   CIK {c.cik}\n{line}")
    print(f"  Value chain : {c.value_chain_position or 'unclassified'}")
    print(f"  Fiscal year : ends month {c.fiscal_year_end or 'n/a'}")
    if profile.as_of:
        print(f"  Data as of  : {profile.as_of}")

    stats = artifacts.stats
    print(f"\n  Retrieval   : {stats.get('filings_retrieved', 0)}"
          f"/{stats.get('filings_selected', 0)} filings"
          f"   network={stats.get('network_requests', 0)}"
          f"  cache_hits={stats.get('cache_hits', 0)}")

    print(f"\n{'-' * 72}\n  RETRIEVED SOURCES\n{'-' * 72}")
    for src in profile.sources:
        status = "FAIL" if src.error else f"{src.content_hash[:8] if src.content_hash else '--------'}"
        kind = src.kind.value
        print(f"  [{status}] {src.source_id:<24} {kind:<12} {src.title[:38]}")

    if artifacts.financials and artifacts.financials.metrics:
        basis = "MOST RECENT QUARTER" if quarterly else "MOST RECENT FISCAL YEAR"
        print(f"\n{'-' * 72}\n  FINANCIAL METRICS (XBRL, deterministic - no LLM)"
              f"\n  Basis: {basis}. YoY is always like-for-like in period length."
              f"\n{'-' * 72}")
        for name, metric in artifacts.financials.metrics.items():
            period = metric.latest_quarter if quarterly else metric.latest_annual
            if period is None:
                period = metric.latest
            if period is None:
                continue
            value = _fmt_money(period.value, metric.unit)
            growth = (
                f"{metric.yoy_growth_pct:+.1f}% YoY"
                if metric.yoy_growth_pct is not None
                else "no YoY"
            )
            print(
                f"  {name:<20} {value:>14}  {growth:>14}   "
                f"[{period.period_label}]  {metric.concept}"
            )
            print(
                f"  {'':<20} {metric.note:<32} accn {period.accession_number}"
            )

    if profile.business_segments:
        print(f"\n{'-' * 72}\n  SEGMENT REVENUE (deterministic table parse from 10-K MD&A)"
              f"\n{'-' * 72}")
        for seg in profile.business_segments:
            rev = f"{seg.revenue_millions:,.0f}M" if seg.revenue_millions else "n/a"
            pct = (
                f"{seg.revenue_pct_of_total:.0f}%"
                if seg.revenue_pct_of_total is not None
                else "n/a"
            )
            chg = (
                f"{seg.revenue_change_pct:+.1f}% YoY"
                if seg.revenue_change_pct is not None
                else ""
            )
            print(f"  {seg.name:<26} {rev:>10}  {pct:>4}  {chg:>12}")
            if seg.revenue_note:
                print(f"  {'':<26} {seg.revenue_note}")

    if profile.geographic_revenue:
        print(f"\n{'-' * 72}\n  GEOGRAPHIC REVENUE (deterministic table parse from 10-K MD&A)"
              f"\n{'-' * 72}")
        for geo in profile.geographic_revenue:
            rev = f"{geo.revenue_millions:,.0f}M" if geo.revenue_millions else "n/a"
            pct = (
                f"{geo.pct_of_total:.0f}%"
                if geo.pct_of_total is not None
                else "n/a"
            )
            chg = (
                f"{geo.change_pct:+.1f}% YoY"
                if geo.change_pct is not None
                else ""
            )
            tag = "  [subtotal - excluded from region sum]" if geo.is_subtotal else ""
            print(f"  {geo.region:<18} {rev:>10}  {pct:>4}  {chg:>12}{tag}")

    if profile.warnings:
        print(f"\n{'-' * 72}\n  WARNINGS ({len(profile.warnings)})\n{'-' * 72}")
        for w in profile.warnings:
            print(f"  ! {w}")

    print()


def print_review(artifacts: pipeline.RunArtifacts, settings: object) -> None:
    """Show exactly what stage 3 would send, without spending a token.

    This is the cheapest possible sanity check on cost: if the chunks and token
    estimate below look wrong, extraction would have been wrong too - and we would
    have paid to discover it. It runs the real planner, not an approximation, so
    the numbers here are the numbers the model would receive.
    """
    from semicon.chunk import find_sections, plan_extraction, select_sections
    from semicon.sources.sec_edgar import html_to_text

    print(f"{'=' * 72}\n  EXTRACTION PREVIEW (real planner, no tokens spent)\n{'=' * 72}")

    raw_chars = 0
    plan = None
    source_id = None

    for fs in artifacts.successful_filings:
        if fs.artifact is None or fs.filing is None or not fs.filing.is_annual:
            continue
        text = html_to_text(fs.artifact.text)
        raw_chars = len(text)
        sections = select_sections(find_sections(text))
        if not sections:
            continue
        plan = plan_extraction(
            sections,
            fs.source.source_id,
            max_chunks=settings.llm.max_extraction_chunks,
            max_total_chars=settings.llm.max_extraction_chars,
        )
        source_id = fs.source.source_id
        break

    if plan is None:
        print("  No annual report text available to plan against.\n")
        return

    print(f"  Source             : {source_id}")
    print(f"  Filing text        : {raw_chars:,} chars  (~{raw_chars // 4:,} tokens if sent whole)")
    print(f"  Chunks available   : {plan.total_chunks_available}")
    print(f"  Duplicates removed : {plan.dropped_duplicates}")
    print(f"  Chunks to send     : {len(plan.chunks)}")
    print(f"  EXTRACTION CONTEXT : {plan.total_chars:,} chars  (~{plan.est_tokens:,} tokens)")
    print(
        f"  Reduction          : "
        f"{100 * (1 - plan.total_chars / max(raw_chars, 1)):.1f}% less text than the whole filing"
    )

    counts: dict[str, int] = {}
    for c in plan.chunks:
        counts[c.section_title] = counts.get(c.section_title, 0) + 1
    print("\n  Sections selected:")
    for title, n in counts.items():
        print(f"    {title[:42]:<44} {n} chunk(s)")
    print()


def print_stage_progress(stage: str, n: int, total: int, chunk: object) -> None:
    """Lightweight progress output for the LLM stages.

    Printed to stderr so that stdout stays clean for piping the summary.
    """
    if stage == "extract":
        label = getattr(chunk, "locator", "")
        print(f"  [extract {n}/{total}] {label}", file=sys.stderr, flush=True)
    elif stage == "analyse":
        print("  [synthesise] reasoning over the claim ledger...", file=sys.stderr, flush=True)


def print_coverage(profile: CompanyProfile) -> None:
    """Show evidence coverage per category.

    This is the balance check on the analysis: a report can be fully cited and
    still be lopsided, which is exactly what happened when 66 claims included
    only 2 about products. Unmet targets are shown as shortfalls rather than
    silently omitted.
    """
    if not profile.coverage:
        return
    line = "-" * 72
    met = sum(1 for c in profile.coverage if c.met)
    print(
        f"\n{line}\n  EVIDENCE COVERAGE ({met}/{len(profile.coverage)} categories at target)"
        f"\n{line}"
    )
    for c in profile.coverage:
        mark = "ok  " if c.met else "GAP "
        bar = "#" * min(c.count, 12)
        label = c.category.replace("_", " ")
        print(f"  {mark} {label:<26} {c.count:>2}/{c.target:<2} {bar}")


def print_validation(validation: object) -> None:
    """Show validation outcome - this is the trust signal for the whole run."""
    if validation is None:
        return
    line = "-" * 72
    print(f"\n{line}\n  VALIDATION\n{line}")
    if validation.ok:
        print("  PASS - every citation resolves and all rules hold")
    else:
        print(f"  FAIL - {len(validation.errors)} error(s)")
    for key, value in validation.stats.items():
        print(f"    {key:<22} {value}")
    for err in validation.errors[:10]:
        print(f"  ERROR: {err}")
    for warn in validation.warnings[:10]:
        print(f"  WARN : {warn}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Semiconductor Company Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python main.py "Applied Materials"\n'
               '  python main.py "Applied Materials" --analyze\n'
               '  python main.py "Applied Materials" --preview',
    )
    parser.add_argument("company", help="Company name or ticker, e.g. 'Applied Materials'")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write company_report.md (default: ./company_report.md)",
    )
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="Where to write company_profile.json (default: data/profiles/<ticker>.json)",
    )
    parser.add_argument(
        "--max-filings", type=int, default=None,
        help="Cap the number of filings retrieved (annual reports are kept first)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Bypass the HTTP cache and re-download",
    )
    parser.add_argument(
        "--quarterly", action="store_true",
        help="Show the most recent quarter instead of the most recent fiscal year",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show extraction-stage text volume and token estimate, then exit",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run the LLM stages (requires OPENAI_API_KEY) and produce the report",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Generate the static HTML site from all saved profiles and exit",
    )
    parser.add_argument(
        "--open", dest="open_browser", action="store_true",
        help="Open the generated HTML site in the default browser",
    )
    # --web regenerates the whole site from every saved profile, so it must not
    # require a company argument. Note the None default: argparse reads sys.argv
    # when argv is None, so the scan must do the same or the flag is missed.
    effective_argv = sys.argv[1:] if argv is None else argv
    if "--web" in effective_argv:
        parser_web = argparse.ArgumentParser(prog="main.py --web")
        parser_web.add_argument("--web", action="store_true")
        parser_web.add_argument("--open", dest="open_browser", action="store_true")
        web_args, _ = parser_web.parse_known_args(effective_argv)
        return _generate_site(open_browser=web_args.open_browser)

    args = parser.parse_args(argv)

    settings = load_settings()

    try:
        profile, artifacts = pipeline.run(
            args.company,
            settings=settings,
            max_filings=args.max_filings,
            force_refresh=args.refresh,
        )
    except ResolutionError as exc:
        print(f"\nResolution failed: {exc}\n", file=sys.stderr)
        return 3
    except RetrievalError as exc:
        print(f"\nRetrieval failed: {exc}\n", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print_summary(profile, artifacts, quarterly=args.quarterly)

    if args.preview:
        print_review(artifacts, settings)
        return 0

    # ---- LLM stages ------------------------------------------------------- #
    if args.analyze:
        if not settings.llm.enabled:
            print(
                "LLM stages are disabled (llm.enabled: false in "
                "config/settings.yaml).",
                file=sys.stderr,
            )
            return 2

        print(f"\n{'-' * 72}\n  RUNNING ANALYSIS (LLM stages)\n{'-' * 72}")
        llm_client = LlmClient(settings)
        try:
            if not llm_client.available:
                print(
                    "No OPENAI_API_KEY found. Set it and re-run:\n"
                    '  export OPENAI_API_KEY="sk-..."\n'
                    "Steps 1-2 above completed without a key and their output has "
                    "been saved.",
                    file=sys.stderr,
                )
                pipeline.write_profile(profile, args.profile or pipeline.default_profile_path(profile))
                return 5

            profile, analysis_warnings = pipeline.run_analysis(
                artifacts,
                profile,
                settings=settings,
                client=llm_client,
                progress=print_stage_progress,
            )
        except BudgetExceeded as exc:
            print(f"\nBudget guard tripped: {exc}\n", file=sys.stderr)
            return 6
        except LlmError as exc:
            print(f"\nModel call failed: {exc}\n", file=sys.stderr)
            return 7
        finally:
            print(f"\n  LLM usage: {llm_client.usage.summary()}", file=sys.stderr)
            for model, stats in llm_client.usage.per_model.items():
                print(
                    f"    {model:<16} {stats['calls']:>3} calls  "
                    f"{stats['input']:>8,.0f} in / {stats['output']:>7,.0f} out  "
                    f"${stats['cost']:.4f}",
                    file=sys.stderr,
                )
            llm_client.close()

        print_validation(artifacts.validation)
        print_coverage(profile)

    # ---- outputs ---------------------------------------------------------- #
    profile_path = args.profile or pipeline.default_profile_path(profile)
    pipeline.write_profile(profile, profile_path)
    print(f"  Wrote {profile_path.relative_to(project_root())}")

    report_path = args.out or (project_root() / "company_report.md")
    report_text = render_report(profile, validation=artifacts.validation)
    render_write(report_text, report_path)
    print(f"  Wrote {report_path.relative_to(project_root())}")

    if not args.analyze:
        print(
            "\n  Note: qualitative sections are empty. Re-run with --analyze to "
            "populate them."
        )

    # Regenerate the static site so the HTML view reflects this run.
    _generate_site(open_browser=args.open_browser, quiet=not args.web)
    return 0


def _generate_site(open_browser: bool = False, quiet: bool = False) -> int:
    """Render the static HTML site from saved profiles."""
    from semicon.web import generate_site

    try:
        out_dir, profiles = generate_site()
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 8

    index = out_dir / "index.html"
    if not quiet:
        print(f"\n  Web: {len(profiles)} company page(s) -> {out_dir.name}/index.html")

    if open_browser:
        import webbrowser

        webbrowser.open(index.resolve().as_uri())
        if not quiet:
            print(f"  Opened {index.resolve().as_uri()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
