"""
System/edge case tests for DataPreprocessor
"""
import os
import pandas as pd
from src.data_preprocessing import DataPreprocessor

# Test loading from a missing file
def test_load_and_split_data_missing_file():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/no_file.csv')
    # Should return None for all outputs
    assert X_train is None
    assert X_test is None
    assert y_train is None
    assert y_test is None

# Test loading from an empty file
def test_load_and_split_data_empty():
    empty_file = 'empty.csv'
    pd.DataFrame().to_csv(empty_file)
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data(empty_file)
    # Should return None or empty for all outputs
    assert X_train is None or len(X_train) == 0
    assert X_test is None or len(X_test) == 0
    assert y_train is None or len(y_train) == 0
    assert y_test is None or len(y_test) == 0
    # Remove test file
    if os.path.exists(empty_file):
        os.remove(empty_file)
