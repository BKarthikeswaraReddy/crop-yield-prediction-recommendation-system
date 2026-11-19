"""
Agricultural Year Utilities
Machine Learning Based Crop Yield Prediction & Recommendation System

Handles agricultural years in the Indian standard format (e.g., '2026-27'),
representing a 12-month agricultural cycle from July 1 to June 30.
Decomposes agricultural years into starting calendar year and ending calendar year.
"""

import re
from typing import Dict, Any, List, Tuple

# Supported future agricultural year forecast horizon (10-year planning horizon)
FUTURE_AGRICULTURAL_YEARS: List[str] = [
    "2026-27",
    "2027-28",
    "2028-29",
    "2029-30",
    "2030-31",
    "2031-32",
    "2032-33",
    "2033-34",
    "2034-35",
    "2035-36",
]


def parse_agricultural_year(year_val: Any) -> Dict[str, Any]:
    """
    Parse an agricultural year string into its calendar components.

    Accepts:
      - Standard agricultural year format: '2026-27' or '2026-2027'
      - Single calendar year: '2026' or 2026 (assumes 2026-27)

    Returns
    -------
    dict with:
      - 'start_year': int, starting calendar year (e.g. 2026)
      - 'end_year': int, ending calendar year (e.g. 2027)
      - 'formatted': str, standard format (e.g. '2026-27')
      - 'display_label': str, descriptive label (e.g. '2026-27 (Jul 2026 – Jun 2027)')
      - 'year_num': float, numerical representation for ML models (e.g. 2026.0)
      - 'is_future': bool, True if start_year >= 2026
    """
    if year_val is None:
        default_start = 2026
        return {
            "start_year": default_start,
            "end_year": default_start + 1,
            "formatted": f"{default_start}-{str(default_start + 1)[-2:]}",
            "display_label": f"{default_start}-{str(default_start + 1)[-2:]} (Jul {default_start} – Jun {default_start + 1})",
            "year_num": float(default_start),
            "is_future": True,
        }

    s = str(year_val).strip()

    # Match patterns like '2026-27', '2026-2027', '2026/27', '2026/2027'
    m_range = re.match(r"^(\d{4})[-/](\d{2,4})$", s)
    if m_range:
        start_year = int(m_range.group(1))
        end_part = m_range.group(2)
        if len(end_part) == 2:
            century = (start_year // 100) * 100
            end_year = century + int(end_part)
            # Handle century wrap if applicable
            if end_year < start_year:
                end_year += 100
        else:
            end_year = int(end_part)
    else:
        # Match single 4-digit year like '2026'
        m_single = re.search(r"(\d{4})", s)
        if m_single:
            start_year = int(m_single.group(1))
            end_year = start_year + 1
        else:
            start_year = 2026
            end_year = 2027

    short_end = str(end_year)[-2:]
    formatted = f"{start_year}-{short_end}"
    display_label = f"{formatted} (Jul {start_year} – Jun {end_year})"

    return {
        "start_year": start_year,
        "end_year": end_year,
        "formatted": formatted,
        "display_label": display_label,
        "year_num": float(start_year),
        "is_future": start_year >= 2026,
    }


def is_valid_future_year(year_val: Any) -> bool:
    """Return True if year_val corresponds to a supported future prediction year."""
    parsed = parse_agricultural_year(year_val)
    return parsed["formatted"] in FUTURE_AGRICULTURAL_YEARS or (2026 <= parsed["start_year"] <= 2040)

