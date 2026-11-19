"""
Authoritative State-District Mapping for India
Machine Learning Based Crop Yield Prediction & Recommendation System

Separates Andhra Pradesh (post-2014 bifurcation) and Telangana correctly.
Historical undivided AP records in the CSV are preserved; only the UI
dropdown is corrected to reflect the present administrative structure.
"""

from typing import List, Optional

# ---------------------------------------------------------------------------
# Post-bifurcation Andhra Pradesh (13 districts in Agmarknet dataset)
# ---------------------------------------------------------------------------
_AP_DISTRICTS = [
    "ANANTAPUR", "CHITTOOR", "EAST GODAVARI", "GUNTUR", "KADAPA",
    "KRISHNA", "KURNOOL", "PRAKASAM", "SPSR NELLORE", "SRIKAKULAM",
    "VISAKHAPATANAM", "VIZIANAGARAM", "WEST GODAVARI"
]

# ---------------------------------------------------------------------------
# Telangana (districts found in the Agmarknet dataset incl. historical AP ones)
# ---------------------------------------------------------------------------
_TELANGANA_DISTRICTS = [
    "ADILABAD", "BHADRADRI", "HANUMAKONDA", "HYDERABAD", "JAGITIAL",
    "JANGOAN", "JAYASHANKAR", "JOGULAMBA", "KAMAREDDY", "KARIMNAGAR",
    "KHAMMAM", "KOMARAM BHEEM ASIFABAD", "MAHABUBABAD", "MAHBUBNAGAR",
    "MANCHERIAL", "MEDAK", "MEDCHAL", "MEDCHAL MALKAJGIRI", "MULUGU",
    "NAGARKURNOOL", "NALGONDA", "NARAYANAPET", "NIRMAL", "NIZAMABAD",
    "PEDDAPALLI", "RAJANNA", "RANGAREDDI", "SANGAREDDY", "SIDDIPET",
    "SURYAPET", "VIKARABAD", "WANAPARTHY", "WARANGAL", "WARANGAL URBAN",
    "YADADRI"
]

# ---------------------------------------------------------------------------
# Districts that appear under "Andhra Pradesh" in the pre-bifurcation CSV
# (needed for backward-compatible backend queries)
# ---------------------------------------------------------------------------
HISTORICAL_UNDIVIDED_AP_DISTRICTS = sorted(set(_AP_DISTRICTS + _TELANGANA_DISTRICTS))

# ---------------------------------------------------------------------------
# Full authoritative state → district mapping
# (only states that have entries in the agricultural dataset are listed;
#  all other states fall back to the raw CSV data)
# ---------------------------------------------------------------------------
AUTHORITATIVE_STATE_DISTRICTS = {
    "Andhra Pradesh": sorted(_AP_DISTRICTS),
    "Telangana": sorted(_TELANGANA_DISTRICTS),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_districts(state: str, fallback_districts: Optional[List[str]] = None) -> List[str]:
    """
    Return the authoritative list of districts for a state.

    For Andhra Pradesh and Telangana the corrected post-bifurcation lists
    are returned.  For every other state the caller-supplied *fallback_districts*
    list (usually derived from the raw CSV) is returned unchanged.

    Parameters
    ----------
    state : str
        State name exactly as it appears in the production dataset.
    fallback_districts : list[str] | None
        Districts from the raw dataset to use when no authoritative override
        exists.  Pass ``None`` (or omit) to get an empty list for unknown states.

    Returns
    -------
    list[str]
        Sorted list of district names (upper-case, matching CSV conventions).
    """
    normalised = str(state).strip()
    if normalised in AUTHORITATIVE_STATE_DISTRICTS:
        return AUTHORITATIVE_STATE_DISTRICTS[normalised]
    return fallback_districts if fallback_districts is not None else []


def validate_state_district(state: str, district: str) -> bool:
    """
    Return True if *district* is valid for *state* according to the
    authoritative mapping.

    For states not in the authoritative map this always returns True (we
    cannot validate them without exhaustive enumeration) so callers should
    treat the result as "not invalidated" rather than "confirmed valid".

    Parameters
    ----------
    state : str
    district : str

    Returns
    -------
    bool
    """
    normalised_state = str(state).strip()
    normalised_district = str(district).strip().upper()

    if normalised_state not in AUTHORITATIVE_STATE_DISTRICTS:
        return True  # cannot validate; pass through

    allowed = [d.upper() for d in AUTHORITATIVE_STATE_DISTRICTS[normalised_state]]
    return normalised_district in allowed


def is_historical_ap_district(district: str) -> bool:
    """
    Return True if *district* appeared in the pre-bifurcation Andhra Pradesh
    dataset (i.e. it is in either modern AP or Telangana).

    Useful for backend queries that need to match historical CSV rows where
    State == 'Andhra Pradesh' regardless of modern administrative boundaries.

    Parameters
    ----------
    district : str

    Returns
    -------
    bool
    """
    normalised = str(district).strip().upper()
    return normalised in [d.upper() for d in HISTORICAL_UNDIVIDED_AP_DISTRICTS]

