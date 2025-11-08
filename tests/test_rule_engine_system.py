"""
System test for RuleEngine:
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from src.rule_engine import RuleEngine  


def _baseline_applicant():
    return {
        "age": 30,
        "annual_income": 30_000,
        "employment_length": 2,
        "credit_score": 700,
        "debt_to_income": 0.20,
        "num_open_accounts": 8,
        "delinquencies_2y": 0,
        "inquiries_6m": 1,
        "loan_amount": 10_000,
        "interest_rate": 0.08,
        "purpose": "personal",
        "home_ownership": "rent",
        "channel": "online",
        "region": "north",
        "loan_term_months": 36,
    }


def test_missing_or_invalid_fields_handling():
    """
    Does the applicant get rejected if one or more fields missing.
    """
    engine = RuleEngine()
    applicant = _baseline_applicant()
    predicted_pd = 0.05 

    applicant.pop("credit_score", None)

    try:
        result = engine.apply_rules(applicant, predicted_pd)
    except TypeError as e:
        pytest.fail(
            f"RuleEngine crashed on missing field: {e}. ")

    assert result["Decision"] == "Rejected"

    reasons = result.get("Reasons", [])
    assert reasons, f"No reasons returned. Got: {result}"

