"""
Integration test for RuleEngine:
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from src.rule_engine import RuleEngine 

def test_ruleengine_end_to_end_decision():
    """
    Checks that using the default thresholds an applicant that violates many rules fails correctly
    """
    engine = RuleEngine() 
    applicant = {
        "age": 30,
        "annual_income": 30_000,
        "employment_length": 2,
        "credit_score": 600,         
        "debt_to_income": 0.40,      
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
    predicted_pd = 0.15

    result = engine.apply_rules(applicant, predicted_pd)

    assert result["Decision"] == "Rejected"
    expected_reasons = {
        "High Probability of Default",
        "Low Credit Score",
        "High Debt-to-Income Ratio",
    }
    assert expected_reasons.issubset(set(result["Reasons"]))
    assert result["Predicted_PD"] == predicted_pd
