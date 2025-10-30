"""
Unit tests for LoanDecisionSystem class
"""
import sys
sys.path.insert(0, './src')
import pytest
from descision_system import LoanDecisionSystem

class DummyModel:
    def predict_proba(self, X):
        return [[0.2, 0.8]] * len(X)

class DummyRuleEngine:
    # Fake rule engine always returning 'Approved'
    def apply_rules(self, applicant_data, pd_value):
        return 'Approved'
    # Fake method to update rules but does nothing
    # kew word args key/val ** puts it into dict - in real-world, would use dict to update rule params
    def update_rules(self, **kwargs):
        pass

# Monkeypatch allows to temp change how parts of code works suring testing
"""
-> Swap out real joblib.load function (which loads actual models from files) with a fake one that pretends to load model (DummyModel/None).

-> Swap out real RuleEngine class with DummyRuleEngine class, so LoanDecisionSystem uses fake.

"""
def test_init_success(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda path: DummyModel())
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    system = LoanDecisionSystem()
    assert system.ml_model is not None
    assert system.rule_engine is not None

def test_make_decision(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda path: DummyModel())
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    system = LoanDecisionSystem()
    applicant = {
        'age': 30, 'annual_income': 50000, 'employment_length': 5, 'credit_score': 700,
        'debt_to_income': 0.2, 'num_open_accounts': 5, 'delinquencies_2y': 0,
        'inquiries_6m': 1, 'loan_amount': 10000, 'interest_rate': 0.05,
        'purpose': 'car', 'home_ownership': 'own', 'channel': 'online',
        'region': 'north', 'loan_term_months': 36
    }
    result = system.make_decision(applicant)
    # Purposely fail: expect 'Rejected' but dummy always returns 'Approved'
    assert result == 'Rejected'

def test_update_rule_parameters(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda path: DummyModel())
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    system = LoanDecisionSystem()
    # Should not raise error
    system.update_rule_parameters(pd_threshold=0.2)
