import pandas as pd
# Omar to impl
from src.ml_model_training import MLModelTrainer
# Omar to impl x2
from src.rule_engine import RuleEngine
from src.data_preprocessing import DataPreprocessor

class LoanDecisionSystem:
    def __init__(self, model_path="logistic_regression_model.joblib"):
        self.ml_model = MLModelTrainer.load_model(model_path)
        if self.ml_model is None:
            print("Model not loaded. Train and save the model first.")
            raise Exception("Model not loaded.")
        self.rule_engine = RuleEngine()

    def make_decision(self, applicant_data):
        # Get feature names from DataPreprocessor
        preprocessor = DataPreprocessor()
        features = preprocessor.numeric_features + preprocessor.categorical_features
        # Make DataFrame for prediction
        df = pd.DataFrame([applicant_data], columns=features)
        # Predict PD
        pd_value = self.ml_model.predict_proba(df)[0][1]
        print("Predicted PD:", round(pd_value, 4))
        # Apply rules
        result = self.rule_engine.apply_rules(applicant_data, pd_value)
        return result

    def update_rule_parameters(self, **kwargs):
        self.rule_engine.update_rules(**kwargs)

if __name__ == "__main__":
    applicant1 = {
        'age': 32,
        'annual_income': 60000,
        'employment_length': 6,
        'credit_score': 690,
        'debt_to_income': 0.30,
        'num_open_accounts': 9,
        'delinquencies_2y': 0,
        'inquiries_6m': 1,
        'loan_amount': 12000,
        'interest_rate': 0.09,
        'purpose': 'debt_consolidation',
        'home_ownership': 'rent',
        'channel': 'online',
        'region': 'north',
        'loan_term_months': 36
    }

    applicant2 = {
        'age': 25,
        'annual_income': 20000,
        'employment_length': 1,
        'credit_score': 550,
        'debt_to_income': 0.55,
        'num_open_accounts': 3,
        'delinquencies_2y': 2,
        'inquiries_6m': 3,
        'loan_amount': 5000,
        'interest_rate': 0.20,
        'purpose': 'personal',
        'home_ownership': 'rent',
        'channel': 'branch',
        'region': 'east',
        'loan_term_months': 12
    }

    try:
        system = LoanDecisionSystem()
        print("\n--- Applicant 1 ---")
        print(system.make_decision(applicant1))
        print("\n--- Applicant 2 ---")
        print(system.make_decision(applicant2))
        print("\n--- Update PD threshold ---")
        system.update_rule_parameters(pd_threshold=0.15)
        print(system.make_decision(applicant1))
    except Exception as e:
        print("Error:", e)
        print("Run ml_model_training.py to save the model.")