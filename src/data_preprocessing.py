import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Class helps prep loan data for ML
class DataPreprocessor:
    def __init__(self, target_col="default_12m"):
        # The column want to predict
        self.target_col = target_col
        # List of columns that are categories
        self.categorical_features = ["purpose", "home_ownership", "channel", "region", "loan_term_months"]
        # List of columns that are numbers
        self.numeric_features = ["age", "annual_income", "employment_length", "credit_score",
                                 "debt_to_income", "num_open_accounts", "delinquencies_2y",
                                 "inquiries_6m", "loan_amount", "interest_rate"]
        # Set up the preprocessor to scale numbers and encode categories
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
        ], remainder='passthrough')

    # Load CSV data and split it into train and test sets
    def load_and_split_data(self, filepath, test_size=0.25, random_state=42):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print("File not found:", filepath)
            return None, None, None, None
        # X is the input features, y is the target
        X = df[self.categorical_features + self.numeric_features]
        y = df[self.target_col].astype(int)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    # Fit the preprocessor on the training data
    def fit_preprocessor(self, X_train):
        self.preprocessor.fit(X_train)
        print("Preprocessor fitted.")

    # Transform the data using the fitted preprocessor
    def transform_data(self, X):
        return self.preprocessor.transform(X)

    # Save the preprocessor to a file
    def save_preprocessor(self, filename="preprocessor.joblib"):
        joblib.dump(self.preprocessor, filename)
        print("Preprocessor saved.")

    # Load a preprocessor from a file
    @staticmethod
    def load_preprocessor(filename="preprocessor.joblib"):
        try:
            return joblib.load(filename)
        except FileNotFoundError:
            print("Preprocessor file not found.")
            return None

if __name__ == "__main__":
    # Run file to test preprocessor
    path = "data/loan_applications.csv"
    dp = DataPreprocessor()
    X_train, X_test, y_train, y_test = dp.load_and_split_data(path)
    if X_train is not None:
        dp.fit_preprocessor(X_train)
        X_train_p = dp.transform_data(X_train)
        X_test_p = dp.transform_data(X_test)
        print("Train shape:", X_train_p.shape)
        print("Test shape:", X_test_p.shape)
        dp.save_preprocessor()