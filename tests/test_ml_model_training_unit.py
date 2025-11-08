"""
Unit tests for MLModelTrainer class
"""
import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

if "data_preprocessing" not in sys.modules:
    shim = types.ModuleType("data_preprocessing")
    class DataPreprocessor: 
        pass
    shim.DataPreprocessor = DataPreprocessor
    sys.modules["data_preprocessing"] = shim

import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.ml_model_training import MLModelTrainer


@pytest.fixture
def test_data_and_preproc():
    df = pd.DataFrame({
        "age": [25, 40, 60, 30, 45, 22, 55, 33],
        "annual_income": [20_000, 60_000, 45_000, 35_000, 80_000, 18_000, 52_000, 29_000],
        "employment_length": [1, 10, 5, 3, 11, 0, 7, 2],
        "credit_score": [550, 720, 690, 660, 800, 510, 700, 640],
        "debt_to_income": [0.55, 0.20, 0.30, 0.40, 0.18, 0.60, 0.25, 0.45],
        "num_open_accounts": [3, 10, 7, 5, 12, 2, 8, 4],
        "delinquencies_2y": [2, 0, 1, 0, 0, 3, 1, 1],
        "inquiries_6m": [3, 1, 2, 1, 0, 4, 1, 2],
        "loan_amount": [5_000, 15_000, 10_000, 8_000, 20_000, 4_000, 12_000, 7_000],
        "interest_rate": [0.20, 0.05, 0.10, 0.15, 0.04, 0.22, 0.08, 0.18],
        "purpose": ["personal","car","home_improvement","debt_consolidation","personal","car","home_improvement","personal"],
        "home_ownership": ["rent","mortgage","own","rent","mortgage","rent","own","rent"],
        "channel": ["online","branch","online","branch","online","branch","online","branch"],
        "region": ["north","east","west","south","north","east","south","west"],
        "loan_term_months": [12, 36, 24, 48, 60, 12, 36, 24],
        "default_12m": [1,0,0,1,0,1,0,1],
    })
    X = df.drop(columns=["default_12m"])
    y = df["default_12m"].astype(int)

    num_cols = ["age","annual_income","credit_score","debt_to_income"]
    cat_cols = ["purpose","region"]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return X, y, preproc


def test_train_correct_pipeline(test_data_and_preproc):
    """
    Pipeline has been trained
    """
    X, y, preproc = test_data_and_preproc
    trainer = MLModelTrainer()
    trainer.train_model(X, y, preproc)

    assert hasattr(trainer, "pipeline")

    from sklearn.linear_model import LogisticRegression
    clf = trainer.pipeline.named_steps["classifier"]
    assert isinstance(clf, LogisticRegression)
    assert clf.max_iter == 1000
    assert clf.random_state == 42
    assert clf.class_weight is not None 

def test_evaluate_model_returns_metrics(test_data_and_preproc):
    """
    Ensure that evaluate model prints all the required variables
    """
    X, y, preproc = test_data_and_preproc
    trainer = MLModelTrainer()
    trainer.train_model(X, y, preproc)

    y_proba, y_pred = trainer.evaluate_model(X, y, pd_cutoff=0.12)

    assert isinstance(y_proba, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_proba.shape == (len(y),)
    assert y_pred.shape == (len(y),)
    assert set(np.unique(y_pred)).issubset({0, 1})

def test_save_and_load_methods(test_data_and_preproc, tmp_path):
    """
    Save_model() write the model load_model() loads and gives same predictions
    """
    X, y, preproc = test_data_and_preproc
    trainer = MLModelTrainer(model_filepath=str(tmp_path / "test_model.joblib"))
    trainer.train_model(X, y, preproc)

    probs_before = trainer.pipeline.predict_proba(X)[:, 1].copy() 
    trainer.save_model()
    assert os.path.exists(trainer.model_filepath)

    loaded_model = MLModelTrainer.load_model(path=trainer.model_filepath)
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict_proba")

    probs_after = loaded_model.predict_proba(X)[:, 1]  
    np.testing.assert_allclose(probs_before, probs_after, rtol=0, atol=1e-12)  #
