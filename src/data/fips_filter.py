"""
FIPS country-code whitelist and filtering utilities
for the GDELT data pipeline.

This module ensures that only recognised sovereign-state
actor codes propagate into the graph, eliminating numeric IDs,
NGOs, and other noisy actor strings.

Author: OpenCode
"""
from typing import FrozenSet, Set

# ── GDELT FIPS actor codes for major powers ──────────────────────────────
# These are the 3-letter codes GDELT uses for its 20 chosen countries.
# Any event whose Actor1Code or Actor2Code is NOT in this set is dropped.
MAJOR_POWERS: FrozenSet[str] = frozenset({
    "USA",   # United States
    "CHN",   # China
    "RUS",   # Russia
    "IND",   # India
    "GBR",   # United Kingdom
    "FRA",   # France
    "DEU",   # Germany
    "JPN",   # Japan
    "ISR",   # Israel
    "IRN",   # Iran
    "PAK",   # Pakistan
    "KOR",   # South Korea
    "AUS",   # Australia
    "BRA",   # Brazil
    "CAN",   # Canada
    "EGY",   # Egypt
    "SAU",   # Saudi Arabia
    "TUR",   # Turkey
    "UKR",   # Ukraine
    "ZAF",   # South Africa
})

# ISO-3166 numeric codes that GDELT *sometimes* uses as Actor codes
# We map them back to alpha-3.
NUMERIC_TO_ALPHA: dict[str, str] = {
    "002": "USA",  # USA (GDELT sometimes encodes this way in GKG exports)
    "100": "USA",
    "160": "USA",
    "200": "CHN",
    "365": "RUS",
    "375": "RUS",
    "700": "IND",
    "732": "IND",
}


def normalize_actor_code(raw: str | None) -> str | None:
    """
    Convert a raw GDELT actor code to a standardised alpha-3 code.

    Steps:
      1. Strip whitespace and uppercase.
      2. If it is a known 3-letter FIPS code → return it.
      3. If it matches a known numeric alias → translate and return.
      4. Otherwise → return None (will be filtered out).
    """
    if raw is None:
        return None
    code = str(raw).strip().upper()
    if code in MAJOR_POWERS:
        return code
    if code in NUMERIC_TO_ALPHA:
        return NUMERIC_TO_ALPHA[code]
    return None


def is_known_country(raw: str | None) -> bool:
    """Quick predicate: is this actor code a recognised country?"""
    return normalize_actor_code(raw) is not None


def filter_known_countries(
    actor_set: Set[str] | frozenset,
) -> frozenset[str]:
    """Given a set of raw actor codes, keep only the recognised countries."""
    return frozenset(
        code for code in actor_set if code in MAJOR_POWERS
    ) | frozenset(
        NUMERIC_TO_ALPHA[code]
        for code in actor_set
        if code in NUMERIC_TO_ALPHA
    )


# Convenience alias for external imports
COUNTRY_WHITELIST = MAJOR_POWERS
COUNTRY_LIST = sorted(MAJOR_POWERS)
