class RuleEngine:
    """This clase implements the rule-based engine to assess credit risk. 
    Parameters:
    - pd_threshold: Probability of default threshold.
    """
    def __init__(self, pd_threshold=0.10, min_age=18, max_age=75, min_income=15000, min_employment_length=1, min_credit_score = 650, debt_to_income_ratio=0.35, max_delinquencies_2y=1):
        self.pd_threshold = pd_threshold
        self.min_age = min_age
        self.max_age = max_age
        self.min_income = min_income
        self.min_employment_length = min_employment_length
        self.min_credit_score = min_credit_score
        self.debt_to_income_ratio = debt_to_income_ratio
        self.max_delinquencies_2y = max_delinquencies_2y
        
        
    def apply_rules(self, applicant_data, predicted_pd):
        """Setting the rules for credit risk assessment."""

        reasons = []
        decision = "Approved"

        if predicted_pd >= self.pd_threshold:
            decision = "Rejected"
            reasons.append("High Probability of Default")

        if applicant_data.get("annual_income") < self.min_income:
            decision = "Rejected"
            reasons.append("Annual Income Too Low")

        if applicant_data.get("age") < self.min_age or applicant_data.get("age") > self.max_age:
            decision = "Rejected"
            reasons.append("Age Out of Range")

        if applicant_data.get("employment_length") < self.min_employment_length:
            decision = "Rejected"
            reasons.append("Insufficient Employment Length")

        if applicant_data.get("credit_score") < self.min_credit_score:
            decision = "Rejected"
            reasons.append("Low Credit Score")

        if applicant_data.get("debt_to_income") > self.debt_to_income_ratio:
            decision = "Rejected"
            reasons.append("High Debt-to-Income Ratio")

        if applicant_data.get("delinquencies_2y") > self.max_delinquencies_2y:
            decision = "Rejected"
            reasons.append("Excessive Recent Delinquencies")

        if not reasons:
            reasons.append("All criteria met")

        return {"Decision": decision, "Reasons": reasons, "Predicted_PD": predicted_pd}
    
    def update_rules(self, pd_threshold=None, min_age=None, max_age=None, min_income=None, min_employment_length=None, min_credit_score=None, debt_to_income_ratio=None, max_delinquencies_2y=None):
        """Update the rule parameters dynamically."""
        if pd_threshold is not None:
            self.pd_threshold = pd_threshold
        if min_age is not None:
            self.min_age = min_age
        if max_age is not None:
            self.max_age = max_age
        if min_income is not None:
            self.min_income = min_income
        if min_employment_length is not None:
            self.min_employment_length = min_employment_length
        if min_credit_score is not None:
            self.min_credit_score = min_credit_score
        if debt_to_income_ratio is not None:
            self.debt_to_income_ratio = debt_to_income_ratio
        if max_delinquencies_2y is not None:
            self.max_delinquencies_2y = max_delinquencies_2y