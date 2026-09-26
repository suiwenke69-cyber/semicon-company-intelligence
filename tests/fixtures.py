"""Test fixtures: verbatim excerpts from real SEC filings.

These strings are copied from AMAT's actual 10-K MD&A text layer (the output of
``sec_edgar.html_to_text``), including its quirks. They are kept short and
attributed, and exist so the parser tests run offline at zero cost.

Using real text matters: every parsing bug found during development came from a
quirk that a hand-written "tidy" fixture would have hidden - a row without
thousands separators, a stray "-" in a label, three-year column layouts, and
wording that changes between filing years.
"""

# FY2025 10-K: two-year layout, "Net revenue ... was as follows".
# The "Europe" row deliberately has NO thousands separator on 962.
SEGMENT_FY2025 = """Net revenue by segment for the periods presented were as follows:

Change
2025 2024 2025 over 2024

(In millions, except percentages)
Semiconductor Systems $ 20,798 73% $ 19,911 73% 4 %
Applied Global Services 6,385 23% 6,225 23% 3 %
Corporate and Other 1,185 4% 1,040 4% 14 %
Total $ 28,368 100% $ 27,176 100% 4 %
"""

# FY2024 10-K: includes the small Display and Corporate and Other segments,
# which a magnitude-based filter would wrongly discard.
SEGMENT_FY2024 = """Net revenue by segment for the periods presented were as follows:

Change
2024 2023 2024 over 2023

(In millions, except percentages)
Semiconductor Systems $ 19,911 73% $ 19,698 74% 1 %
Applied Global Services 6,225 23% 5,732 22% 9 %
Display 885 3% 868 3% 2 %
Corporate and Other 155 1% 219 1% (29) %
Total $ 27,176 100% $ 26,517 100% 2 %
"""

# FY2023 10-K: different wording ("Net sales ... were as follows") and a
# three-year layout with two change columns.
GEO_FY2023 = """Net sales by geographic region, determined by the location of customers' facilities to which products were shipped, were as follows:

Change
2023 2022 2021 2023 over 2022 2022 over 2021

(In millions, except percentages)
China $ 7,247 27% $ 7,254 28% $ 7,535 33% - % (4) %
Korea 4,609 18% 4,395 17% 5,012 22% 5 % (12) %
Taiwan 5,670 21% 6,262 24% 4,742 20% (9) % 32 %
Japan 2,075 8% 2,012 8% 1,962 8% 3 % 3 %
Southeast Asia 758 3% 1,084 4% 677 3% (30) % 60 %
Asia Pacific 20,359 77% 21,007 81% 19,928 86% (3) % 5 %
United States 4,006 15% 3,104 12% 2,038 9% 29 % 52 %
Europe 2,152 8% 1,674 7% 1,097 5% 29 % 53 %
Total $ 26,517 100% $ 25,785 100% $ 23,063 100% 3 % 12 %
"""

# FY2025 10-K geographic table: two-year layout with an "Asia Pacific" subtotal
# that must be excluded from any region sum.
GEO_FY2025 = """Net revenue by geographic region, determined by the location of customers' facilities to which products were shipped and services were performed, was as follows:

Change
2025 2024 2025 over 2024

(In millions, except percentages)
China $ 8,529 30% $ 10,117 37% (16) %
Korea 5,608 20% 4,493 17% 25 %
Taiwan 6,857 24% 4,010 15% 71 %
Japan 2,273 8% 2,154 8% 6 %
Southeast Asia 1,076 4% 1,141 4% (6) %
Asia Pacific 24,343 86% 21,915 81% 11 %
United States 3,063 11% 3,818 14% (20) %
Europe 962 3% 1,443 5% (33) %
Total $ 28,368 100% $ 27,176 100% 4 %
"""

# The column-header line that must never be parsed as a data row. It previously
# parsed as a row labelled "over" with four amounts, which corrupted the
# parts-sum-to-total check and caused correct rows to be discarded.
HEADER_ROW = "2025 2024 2025 over 2024"

# Rows that must be rejected as non-data.
NON_DATA_ROWS = [
    HEADER_ROW,
    "(In millions, except percentages)",
    "Change",
    "2025 2024",
    "Table of Contents",
]


# AMAT's BACKLOG table. It is introduced by a sentence containing both "revenue"
# and "segment", so an over-broad lead-in regex matched it and reported backlog as
# segment revenue - silently shifting every segment figure ($7,105M instead of
# $20,798M). It must never be parsed as revenue.
BACKLOG_SEGMENT = """Backlog by reportable segment as of October 26, 2025 and October 27, 2024 was as follows:

2025 2024

(In millions, except percentages)
Semiconductor Systems $ 7,105 47% $ 6,420 47%
Applied Global Services 7,141 48% 6,520 48%
Corporate and Other 756 5% 690 5%
Total $ 15,002 100% $ 13,630 100%
"""

# The sentence that fooled the loose pattern: it mentions revenue and segment but
# introduces backlog, not revenue.
BACKLOG_PREAMBLE = (
    "Deferred revenue has not been recognized; and (2) contractual service revenue "
    "and maintenance fees. Backlog by reportable segment as of October 26, 2025 was "
    "as follows:"
)
