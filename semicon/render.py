"""Stage 6: render the structured profile into ``company_report.md``.

Rendering is pure Python. No model is involved, which means the report cannot
introduce a claim that is not already in the profile - the narrative sections are
either backed by a validated claim or visibly marked as empty.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import project_root
from .formatting import format_money, format_percent
from .models import CompanyProfile
from .validate import ValidationReport


def _money(value: float | None, unit: str | None = None) -> str:
    return format_money(value, unit)


def _pct(value: float | None) -> str:
    return format_percent(value)


def _signed_pct(value: float | None) -> str:
    return format_percent(value, signed=True)


def _millions(value: float | None) -> str:
    """Render a figure already denominated in millions.

    The segment and geographic tables are parsed from MD&A tables whose column
    headers say "In millions". Running those through the generic money formatter
    produced "20.8K" for $20,798M, which is technically a correct conversion but
    actively harder to read and inconsistent with the filing it came from.
    """
    if value is None:
        return "n/a"
    return f"{value:,.0f}M"


def build_environment(template_dir: Path | None = None) -> Environment:
    """Jinja environment with the report's formatting filters registered.

    ``StrictUndefined`` is deliberate: a typo in a template variable should fail
    loudly during development rather than silently render as an empty string in a
    report someone might trust.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_dir or project_root() / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["money"] = _money
    env.filters["pct"] = _pct
    env.filters["signed_pct"] = _signed_pct
    env.filters["millions"] = _millions
    return env


def render_report(
    profile: CompanyProfile,
    *,
    validation: ValidationReport | None = None,
    template_name: str = "report.md.j2",
    template_dir: Path | None = None,
) -> str:
    """Render the markdown report for a profile."""
    env = build_environment(template_dir)
    template = env.get_template(template_name)
    return template.render(profile=profile, validation=validation)


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
