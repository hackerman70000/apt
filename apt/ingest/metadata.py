import re
from typing import Set

APT_PATTERNS = [
    r'\bAPT\s*\d+\b',
    r'\bAPT[-_]?\d+\b',
    r'\b(?:Lazarus|Kimsuky|OceanLotus|Fancy Bear|Cozy Bear|Turla|Equation|Carbanak)\b',
]

TECHNIQUE_PATTERN = r'\bT\d{4}(?:\.\d{3})?\b'

def extract_apt_mentions(text: str) -> Set[str]:
    apt_groups = set()

    for pattern in APT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        apt_groups.update(matches)

    return {apt.upper().replace(" ", "") for apt in apt_groups}

def extract_technique_mentions(text: str) -> Set[str]:
    techniques = set(re.findall(TECHNIQUE_PATTERN, text))
    return techniques
