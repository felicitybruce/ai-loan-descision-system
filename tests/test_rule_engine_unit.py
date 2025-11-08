"""
Unit tests for RuleEngine
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from src.rule_engine import RuleEngine  # noqa: E402

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


def test_valid_applicant_approved():
    """
    Verify that an applicant with values in the threshold is approved
    """
    engine = RuleEngine() 
    applicant = _baseline_applicant()
    pd_value = 0.05 

    result = engine.apply_rules(applicant, pd_value)

    assert result["Decision"] == "Approved"
    assert result["Reasons"] == ["All criteria met"]
    assert result["Predicted_PD"] == pd_value


def test_high_pd_rejected():
    """
    Verify when applicant pd is above the threshold is rejected
    """
    engine = RuleEngine()
    applicant = _baseline_applicant()
    pd_value = 0.15 

    result = engine.apply_rules(applicant, pd_value)

    assert result["Decision"] == "Rejected"
    assert "High Probability of Default" in result["Reasons"]
    assert len(result["Reasons"]) == 1


def test_update_rule_thresholds():
    """
    Ensures that rules are correctly updated after the parameters are changed
    """
    engine = RuleEngine()
    applicant = _baseline_applicant()
    applicant["credit_score"] = 640 
    pd_value = 0.08  

    result_before = engine.apply_rules(applicant, pd_value)
    assert result_before["Decision"] == "Rejected"
    assert "Low Credit Score" in result_before["Reasons"]

    engine.update_rules(min_credit_score=600)

    result_after = engine.apply_rules(applicant, pd_value)
    assert result_after["Decision"] == "Approved"
    assert result_after["Reasons"] == ["All criteria met"]
