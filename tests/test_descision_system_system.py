"""
System/edge case tests for LoanDecisionSystem
"""
import sys
sys.path.insert(0, './src')
import pytest
from descision_system import LoanDecisionSystem

class DummyModel:
    def predict_proba(self, X):
        # Ficticuous, always predicts 50/50 chance of default
        return [[0.5, 0.5]] * len(X)

class DummyRuleEngine:
    # Fake rule engine always returning 'Approved'
    def apply_rules(self, applicant_data, pd_value):
        return 'Approved'
    def update_rules(self, **kwargs):
        pass

# Checks what happens if ML model file missing
def test_missing_model(monkeypatch):
    #  Make .joblib file pretend it couldnt find/load modle and return None
    monkeypatch.setattr('joblib.load', lambda path: None)
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    # Expect error when tryign to create LoanDescisionSys
    with pytest.raises(Exception):
        LoanDecisionSystem()

# Test for passing bad applicatn data
def test_invalid_applicant(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda path: DummyModel())
    monkeypatch.setattr('descision_system.RuleEngine', lambda: DummyRuleEngine())
    system = LoanDecisionSystem()
    # Missing required fields
    applicant = {'age': 30}
    # Expect err
    with pytest.raises(Exception):
        system.make_decision(applicant)
