"""
Event-condition parser.

Turns a natural-language condition like:
    "Fed funds rate above 5% by December"
    "SPX below 5000 by 2026-12-18"
    "AAPL between 200 and 240 by Jan 2027"
    "Gold above 2500 by year end"
into a structured EventSpec: (direction, threshold(s), target_date).

The underlying can be provided separately (a Bloomberg ticker or a preset
name) or embedded in the text; the API accepts both. This parser is
deliberately rule-based and transparent -- no LLM -- so results are auditable
by a strategist and reproducible.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EventSpec:
    direction: str                 # 'above' | 'below' | 'between'
    threshold: float               # primary threshold
    threshold_hi: Optional[float]  # for 'between'
    target_date: dt.date
    raw: str
    is_percent: bool = False       # threshold expressed as a percent (rates)

    def describe(self) -> str:
        d = self.target_date.strftime("%d %b %Y")
        unit = "%" if self.is_percent else ""
        if self.direction == "between":
            return f"between {self.threshold}{unit} and {self.threshold_hi}{unit} by {d}"
        return f"{self.direction} {self.threshold}{unit} by {d}"


_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

_ABOVE = re.compile(r"\b(above|over|greater than|more than|exceed(?:s|ing)?|>=?|higher than)\b", re.I)
_BELOW = re.compile(r"\b(below|under|less than|lower than|<=?|beneath)\b", re.I)
_BETWEEN = re.compile(r"\bbetween\b", re.I)

_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%)?")


def _next_occurrence_of_month(month: int, today: dt.date, day: int | None = None) -> dt.date:
    """Return the next future date for the given month (this year if still
    ahead, else next year). If day is None, use month end."""
    year = today.year
    d = day or calendar.monthrange(year, month)[1]
    cand = dt.date(year, month, min(d, calendar.monthrange(year, month)[1]))
    if cand <= today:
        year += 1
        d2 = day or calendar.monthrange(year, month)[1]
        cand = dt.date(year, month, min(d2, calendar.monthrange(year, month)[1]))
    return cand


def parse_target_date(text: str, today: Optional[dt.date] = None) -> dt.date:
    today = today or dt.date.today()
    t = text.lower()

    # ISO date  YYYY-MM-DD
    iso = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if iso:
        return dt.date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    # "year end" / "end of year" / "eoy"
    if re.search(r"\b(year[\s-]?end|end of (the )?year|eoy|year's end)\b", t):
        return dt.date(today.year, 12, 31)

    # "end of <month>" or "<month> <year>" or bare "<month>"
    # month + explicit year
    m = re.search(r"\b(" + "|".join(_MONTHS.keys()) + r")\b\.?\s*(\d{4})", t)
    if m:
        month = _MONTHS[m.group(1)]
        year = int(m.group(2))
        day = calendar.monthrange(year, month)[1]
        return dt.date(year, month, day)

    # bare month name -> next occurrence, month end
    m = re.search(r"\b(" + "|".join(_MONTHS.keys()) + r")\b", t)
    if m:
        return _next_occurrence_of_month(_MONTHS[m.group(1)], today)

    # "in N months" / "N months"
    m = re.search(r"in\s+(\d+)\s+month", t)
    if m:
        n = int(m.group(1))
        month = (today.month - 1 + n) % 12 + 1
        year = today.year + (today.month - 1 + n) // 12
        day = calendar.monthrange(year, month)[1]
        return dt.date(year, month, day)

    # "next quarter end" fallback -> ~3 months out
    if "quarter" in t:
        month = (today.month - 1 + 3) % 12 + 1
        year = today.year + (today.month - 1 + 3) // 12
        return dt.date(year, month, calendar.monthrange(year, month)[1])

    # default: 3 months out
    month = (today.month - 1 + 3) % 12 + 1
    year = today.year + (today.month - 1 + 3) // 12
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def parse_condition(text: str, today: Optional[dt.date] = None,
                    force_percent: Optional[bool] = None) -> EventSpec:
    """Parse a full condition string into an EventSpec."""
    today = today or dt.date.today()
    target = parse_target_date(text, today)

    # Extract the numeric threshold(s), ignoring any year in a date.
    # Remove ISO dates and 4-digit years so they aren't picked up as thresholds.
    scrub = re.sub(r"\d{4}-\d{1,2}-\d{1,2}", " ", text)
    scrub = re.sub(r"\b(19|20)\d{2}\b", " ", scrub)

    nums = _NUM.findall(scrub)
    values = [(float(v), bool(pct)) for v, pct in nums]

    if _BETWEEN.search(text) and len(values) >= 2:
        lo, hi = sorted([values[0][0], values[1][0]])
        is_pct = values[0][1] or values[1][1]
        return EventSpec("between", lo, hi, target, text,
                         is_percent=force_percent if force_percent is not None else is_pct)

    if not values:
        raise ValueError(f"No numeric threshold found in condition: {text!r}")

    thr, is_pct = values[0]
    if force_percent is not None:
        is_pct = force_percent

    if _BELOW.search(text):
        direction = "below"
    else:
        # default to 'above' if 'above' matched or nothing matched
        direction = "above"

    return EventSpec(direction, thr, None, target, text, is_percent=is_pct)


def compute_probability(rnd, spec: EventSpec) -> dict:
    """Given a RiskNeutralDensity and an EventSpec, return the probability plus
    context."""
    if spec.direction == "above":
        p = rnd.prob_above(spec.threshold)
    elif spec.direction == "below":
        p = rnd.prob_below(spec.threshold)
    else:
        p = rnd.prob_between(spec.threshold, spec.threshold_hi)
    return {
        "probability": p,
        "condition": spec.describe(),
        "direction": spec.direction,
        "threshold": spec.threshold,
        "threshold_hi": spec.threshold_hi,
        "target_date": spec.target_date.isoformat(),
    }
