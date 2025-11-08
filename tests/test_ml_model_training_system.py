"""
System tests for MLModelTrainer:
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from src.ml_model_training import MLModelTrainer


def test_evaluate_method_before_training():
    """
    Test that calling evaluate_model before training raises an error.
    """
    trainer = MLModelTrainer()
    with pytest.raises(ValueError) as exc:
        trainer.evaluate_model(X_test=[], y_test=[])
    assert "Pipeline has not been trained yet" in str(exc.value)


def test_save_method_before_training():
    """
    Test that calling save_model before training raises an error.
    """
    trainer = MLModelTrainer()
    with pytest.raises(ValueError) as exc:
        trainer.save_model()
    assert "Pipeline has not been trained yet" in str(exc.value)
