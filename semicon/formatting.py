"""Display formatting shared by the CLI and the report renderer.

Exists because the money formatting was previously implemented twice - once in
``main.py`` and once in ``render.py`` - and the per-share special case was fixed
in one and forgotten in the other, so the report printed an EPS of 8.66 as "9".

Any display rule that more than one surface needs belongs here.
"""

from __future__ import annotations


def format_money(value: float | None, unit: str | None = None) -> str:
    """Format a raw figure compactly, honouring its unit.

    Per-share values are never scaled or rounded to whole units: an EPS of 8.66
    rendered as "9" is simply wrong. The decision is driven by the unit supplied
    with the metric rather than by guessing from magnitude.
    """
    if value is None:
        return "n/a"
    if unit == "USD/shares":
        return f"${value:,.2f}"
    if unit == "shares":
        return f"{value / 1e6:,.1f}M shares"
    magnitude = abs(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            return f"{value / scale:,.1f}{suffix}"
    return f"{value:,.0f}"


def format_percent(value: float | None, *, signed: bool = False) -> str:
    """Format a percentage, optionally with an explicit sign for changes."""
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.0f}%"
