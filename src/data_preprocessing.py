import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

class DataPreprocessor:
  # Default within 12 months y/n
   def __init__(self, target_col="default_12m"):
        # Column name want to predict (e.g., if a loan will default)
        self.target_col = target_col
        # Ficticious categories
        self.categorical_features = ["purpose", "home_ownership", "channel", "region", "loan_term_months"]
        # List of columns
        self.numeric_features = ["age", "annual_income", "employment_length", "credit_score",
                                 "debt_to_income", "num_open_accounts", "delinquencies_2y",
                                 "inquiries_6m", "loan_amount", "interest_rate"]
        # To clean
        self.preprocessor = self._build_preprocessor()

def _build_preprocessor(self):
        """Builds column transformer for preprocessing happenings"""
        return ColumnTransformer(
            transformers=[
                #  Scales to numbers so ML works better
                ("num", StandardScaler(), self.numeric_features),
                # Categories into 0/1s
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ],
            # If not num/cat, just include them in output
            remainder='passthrough'
        )        

def load_and_split_data(self, filepath, test_size=0.25, random_state=42):
        """Loads data and splits it into training & testing """
        try:
            # Read data into dataframe
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: Dataset not found at {filepath}")
            return None, None, None, None

        # Select inputs using num/cat columns
        X = df[self.categorical_features + self.numeric_features]
        # Select output column and make sure its 0/1s
        y = df[self.target_col].astype(int)

        """
        Data in 4 prts:
        - training inputs (X_train), 
        - test inputs (X_test),
        - training outputs (y_train), 
        - test outputs (y_test)
        """
        # 'test_size' how much data to hold back for testing
        # 'random_state' split is same every run
        # 'stratify' both training and test sets have similar proportions of 'y' values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
