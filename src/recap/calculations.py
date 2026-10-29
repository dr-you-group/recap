"""One deliberately narrow deterministic numerical check."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pydantic import Field
from .contracts import Contract


class PercentageInput(Contract):
    numerator: str
    denominator: str
    reported_percent: str
    decimal_places: int = Field(ge=0, le=10)
    same_population_verified: bool
    same_timepoint_verified: bool
    simple_proportion_verified: bool


def check_reported_percentage(values: PercentageInput) -> dict:
    if not all((values.same_population_verified, values.same_timepoint_verified,
                values.simple_proportion_verified)):
        return {"status": "not_applicable", "reason": "unverified_conditions"}
    try:
        n, d, reported = map(Decimal, (values.numerator, values.denominator, values.reported_percent))
        if not all(x.is_finite() for x in (n, d, reported)) or d <= 0 or n < 0 or n > d:
            return {"status": "not_applicable", "reason": "invalid_counts"}
        if n != n.to_integral() or d != d.to_integral():
            return {"status": "not_applicable", "reason": "counts_not_integers"}
        quantum = Decimal(1).scaleb(-values.decimal_places)
        expected = (100 * n / d).quantize(quantum, rounding=ROUND_HALF_UP)
        return {"status": "consistent" if expected == reported else "inconsistent",
                "expected_percent": str(expected), "reported_percent": str(reported),
                "rounding": "ROUND_HALF_UP", "epistemic_status": "calculated"}
    except (InvalidOperation, ValueError):
        return {"status": "not_applicable", "reason": "invalid_number"}
