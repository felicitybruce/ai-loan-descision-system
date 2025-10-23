import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import brier_score_loss
import joblib

from data_preprocessing import DataPreprocessor
from sklearn.pipeline import Pipeline

class MLModelTrainer:
    def __init__(self, model_filepath="models/logistic_regression_model.joblib"):
        self.model = None
        self.model_filepath = model_filepath

    def train_model(self, X_train, y_train, preprocessor):
        """Trains a logistic regression model with class weighting."""
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        self.model = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.model)
        ])

        self.pipeline.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test, pd_cutoff=0.12):
        if not hasattr(self, 'pipeline'):
            raise ValueError("Pipeline has not been trained yet, call train_model() first")
        
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= pd_cutoff).astype(int)

        print("\n--- Model Evaluation ---")
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print("PR-AUC :", average_precision_score(y_test, y_proba))
        print(f"\nConfusion Matrix (PD ≥ {pd_cutoff} → Reject):\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))
        print("Brier Score:", brier_score_loss(y_test, y_proba))

        return y_proba, y_pred
    
    def save_model(self):
        if not hasattr(self, 'pipeline'):
            raise ValueError("Pipeline has not been trained yet, call train_model() first")
        joblib.dump(self.pipeline, self.model_filepath)

    def load_model(path="models/logistic_regression_model.joblib"):
        return joblib.load(path)
    



if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    DATA_PATH = "data/loan_applications.csv"
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data(DATA_PATH)

    if X_train is not None:
        preproc = dp.preprocessor
        trainer = MLModelTrainer()
        trainer.train_model(X_train, y_train, preproc)
        trainer.evaluate_model(X_test, y_test, pd_cutoff=0.12)
        trainer.save_model()
