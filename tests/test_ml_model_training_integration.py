"""
Integration test for MLModelTrainer class
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from src.ml_model_training import MLModelTrainer
from src.data_preprocessing import DataPreprocessor


def test_full_training_pipeline(tmp_path):
    """
    End-to-end test which first loads the data trains the model using preprocessing, evaluates the model and finally saves it successfully
    """
    data_path = PROJECT_ROOT / "data" / "loan_applications.csv"
    if not data_path.exists():
        pytest.skip(f"Sample dataset not found at {data_path}")

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "logistic_regression_model.joblib"

    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data(str(data_path))
    assert X_train is not None and len(X_train) > 0, "Training split is empty"
    assert X_test is not None and len(X_test) > 0, "Test split is empty"
    preproc = dp.preprocessor

    trainer = MLModelTrainer(model_filepath=str(model_file))
    trainer.train_model(X_train, y_train, preproc)

    y_proba, y_pred = trainer.evaluate_model(X_test, y_test, pd_cutoff=0.12)
    assert isinstance(y_proba, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_proba.shape[0] == len(y_test)
    assert y_pred.shape[0] == len(y_test)
    assert np.all((y_proba >= 0.0) & (y_proba <= 1.0))
    assert set(np.unique(y_pred)).issubset({0, 1})

    trainer.save_model()
    assert model_file.exists(), "Model file was not created"
