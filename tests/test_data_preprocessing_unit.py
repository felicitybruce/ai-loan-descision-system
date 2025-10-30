"""
Unit tests for DataPreprocessor class
"""
import os
from src.data_preprocessing import DataPreprocessor

# Test loading and splitting data
def test_load_and_split_data():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/loan_applications.csv')
    # Check that data is loaded
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    # Check that train and test sets are not empty
    assert len(X_train) > 0
    assert len(X_test) > 0

# Test fitting and transforming data
def test_fit_and_transform():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/loan_applications.csv')
    dp.fit_preprocessor(X_train)
    X_train_p = dp.transform_data(X_train)
    X_test_p = dp.transform_data(X_test)
    # Check that transformed data has same number of rows
    assert X_train_p.shape[0] == X_train.shape[0]
    assert X_test_p.shape[0] == X_test.shape[0]

# Test saving and loading preprocessor
def test_save_and_load_preprocessor():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/loan_applications.csv')
    dp.fit_preprocessor(X_train)
    dp.save_preprocessor('preprocessor_test.joblib')
    loaded = DataPreprocessor.load_preprocessor('preprocessor_test.joblib')
    # Check that preprocessor loads
    assert loaded is not None
    # Remove test file
    if os.path.exists('preprocessor_test.joblib'):
        os.remove('preprocessor_test.joblib')

# Test transforming data before fitting
def test_transform_data_without_fit():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/loan_applications.csv')
    # Try to transform without fitting
    error = False
    try:
        dp.transform_data(X_train)
    except Exception:
        error = True
    assert error
