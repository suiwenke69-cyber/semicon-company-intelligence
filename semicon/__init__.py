"""Semiconductor Company Intelligence Agent.

Pipeline, in order:

    resolve  ->  retrieve  ->  [extract]  ->  [analyze]  ->  [render]

Only the bracketed stages use an LLM. Everything else is deterministic Python.
See ``semicon/pipeline.py`` for the orchestration and ``semicon/models.py`` for
the data contracts.
"""

__version__ = "0.1.0"
