"""
Update this file to add more supported languages and styles.
"""

from typing import Literal


LangCode = Literal["fr", "ja", "zh"]
StyleCode = Literal["pirate"]
LangOrStyleCode = LangCode | StyleCode

LANG_MAPPING: dict[LangCode, str] = {
    "fr": "French",
    "ja": "Japanese",
    "zh": "Simplified Chinese",
}
STYLE_MAPPING: dict[StyleCode, str] = {
    "pirate": "pirate",
}
LANG_OR_STYLE_MAPPING: dict[LangOrStyleCode, str] = {**LANG_MAPPING, **STYLE_MAPPING}
