"""SEC EDGAR source adapters.

Two responsibilities, deliberately separated:

* :mod:`sec_edgar`      - discover and download filings (documents).
* :mod:`sec_financials` - parse XBRL into exact numbers (no documents, no LLM).

Both are CIK-driven rather than company-specific, which is what makes adding a
new company a configuration change rather than a code change.
"""

from . import sec_edgar, sec_financials  # noqa: F401
