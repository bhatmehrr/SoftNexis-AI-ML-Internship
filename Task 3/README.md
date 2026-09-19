# Project 1: Student Performance Predictor

Predict a student's exam score based on hours studied.


## Task 3: Evaluate and Make Predictions

**Goal:** Use the trained model to predict scores on the test data,
compare them with actual scores, and calculate the Mean Absolute
Error (MAE) — a metric for how far off the predictions are on average.

### What this script does
- Rebuilds and retrains the model (same as Tasks 1 & 2)
- Predicts scores on the held-out test set
- Calculates and prints the Mean Absolute Error
- Prints a table comparing actual vs predicted scores
- Predicts the score for a new student who studied 4.5 hours

### How to run

```bash
pip install -r requirements.txt
python task3_evaluate_and_predict.py
```

### Tools
- Python
- pandas
- scikit-learn

### Expected output
MAE should be small (around 1–3 points) since this is a clean,
near-linear dataset. The 4.5-hour prediction should land roughly
in the 58–60 range.

### Next steps
Future tasks may include visualizing the regression line with
matplotlib and testing the model on a larger, real-world dataset.