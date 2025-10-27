"""
Integration test for LoanDecisionSystem 
"""
import sys
sys.path.insert(0, './src')
import pytest
from descision_system import LoanDecisionSystem

class DummyModel:
    def predict_proba(self, X):
        # Always predicts high PD
        return [[0.1, 0.9]] * len(X)

# Fake rule engine approves if PD > 0.5, else rejects
class DummyRuleEngine:
    def apply_rules(self, applicant_data, pd_value):
        return 'Approved' if pd_value > 0.5 else 'Rejected'
    # Fake method to update rules but does not do anyhting
    def update_rules(self, **kwargs):
        pass

# Test whole system works together
def test_integration_workflow(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda path: DummyModel())
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    system = LoanDecisionSystem()
    applicant = {
        'age': 40, 'annual_income': 80000, 'employment_length': 10, 'credit_score': 750,
        'debt_to_income': 0.15, 'num_open_accounts': 8, 'delinquencies_2y': 0,
        'inquiries_6m': 2, 'loan_amount': 15000, 'interest_rate': 0.07,
        'purpose': 'home_improvement', 'home_ownership': 'own', 'channel': 'online',
        'region': 'west', 'loan_term_months': 48
    }

    # Will fail - DummyModel gives 0.9 PD. DummyRuleEngine (0.9 > 0.5) says 'Approved'.
    result = system.make_decision(applicant)
    assert result == 'Rejected'
    # Update rule parameter (but DummyRuleEngine ignores it)
    system.update_rule_parameters(pd_threshold=0.8)
    result2 = system.make_decision(applicant)
    # Fail - expect 'Rejected' but dummy always returns 'Approved'
    assert result2 == 'Rejected'
