"""Conservative extraction of number tokens from learner-facing text."""

from dataclasses import dataclass
from fractions import Fraction
import re


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: Fraction
    kind: str
    start: int
    end: int
    decimals: int


_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+"
_TOKEN_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_NUMBER})"
    rf"(?:\s*/\s*(?P<den>\d+))?"
    rf"(?P<percent>\s*%|\s+percent\b)?(?!\w|\.\d)",
    re.IGNORECASE,
)


def extract_numbers(text):
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        raw_num = match.group("num")
        numerator_text = raw_num.replace(",", "")
        denominator_text = match.group("den")
        percent_suffix = match.group("percent")
        if denominator_text is not None:
            numerator = Fraction(numerator_text)
            denominator = int(denominator_text)
            if denominator == 0:
                continue
            value = numerator / denominator
            kind = "fraction"
            decimals = 0
        else:
            decimals = len(numerator_text.split(".", 1)[1]) if "." in numerator_text else 0
            value = Fraction(numerator_text)
            kind = "decimal" if decimals else "int"
            if percent_suffix:
                value /= 100
                kind = "percent"
        tokens.append(
            NumberToken(
                raw=match.group(0),
                value=value,
                kind=kind,
                start=match.start(),
                end=match.end(),
                decimals=decimals,
            )
        )
    return tokens
