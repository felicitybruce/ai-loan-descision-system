"""
Integration test for DataPreprocessor
"""
import os
from src.data_preprocessing import DataPreprocessor

# Test the full workflow: load, fit, transform, save, reload
def test_full_preprocessing_workflow():
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data('data/loan_applications.csv')
    # Check that data is loaded
    assert X_train is not None and X_test is not None
    # Fit preprocessor
    dp.fit_preprocessor(X_train)
    # Transform train and test data
    X_train_p = dp.transform_data(X_train)
    X_test_p = dp.transform_data(X_test)
    # Save preprocessor
    dp.save_preprocessor('preprocessor_test.joblib')
    # Load preprocessor
    loaded_preprocessor = DataPreprocessor.load_preprocessor('preprocessor_test.joblib')
    assert loaded_preprocessor is not None
    # Transform test data again
    X_test_p2 = loaded_preprocessor.transform(X_test)
    # Check that output is the same
    assert (X_test_p2 == X_test_p).all()
    # Remove test file
    if os.path.exists('preprocessor_test.joblib'):
        os.remove('preprocessor_test.joblib')
