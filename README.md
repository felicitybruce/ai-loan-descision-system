# AI Loan Decision System Prototype

This project is a prototype for an AI-powered loan decision system. It uses machine learning and Symbolic rule-based logic to help automate loan approvals/denials.

## Features

- Data preprocessing for loan applications.
- Machine learning model training and prediction.
- Rule engine for business logic.
- Unit, integration, and system/edge case tests.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   - Place your loan application data in `data/loan_applications.csv`.
3. **Train the model:**
   - Run the training script to create the model file:
     ```bash
     python3 src/ml_model_training.py
     ```

## Running the Prototype

To run the main loan decision system and see example decisions:

```bash
python3 src/descision_system.py
```

## Running Tests

To run all tests:

```bash
PYTHONPATH=. pytest tests/
```

To run specific test file:

```bash
PYTHONPATH=. pytest tests/test_data_preprocessing_integration.py
```

## Notes

- If you see errors about missing models, run the training script first.
