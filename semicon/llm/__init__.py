"""LLM access layer.

Deliberately thin. ``client.py`` is the only module in the project that knows
which provider is in use; everything else depends on ``complete_json`` and
``complete_text``.

It is a package rather than a module so that adding a second provider means
adding a file here, not editing the pipeline.
"""

from .client import BudgetExceeded, LlmClient, LlmError, Usage, extract_json

__all__ = ["LlmClient", "LlmError", "BudgetExceeded", "Usage", "extract_json"]
