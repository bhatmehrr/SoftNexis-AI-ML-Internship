# Project 1: Student Performance Predictor

Predict a student's exam score based on hours studied.


## Task 2: Train a Linear Regression Model

**Goal:** Split the data into training and testing sets, then train a
linear regression model that maps hours studied to exam score.

### What this script does
- Recreates the dataset from Task 1
- Splits it into training (80%) and testing (20%) sets
- Trains a `LinearRegression` model
- Prints the learned slope and intercept
- Predicts the score for a sample number of hours studied

### How to run

```bash
pip install -r requirements.txt
python task2_train_linear_regression.py
```

### Tools
- Python
- pandas
- scikit-learn