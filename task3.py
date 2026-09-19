"""
Project 1: Student Performance Predictor
Task 3: Evaluate and Make Predictions

Goal: Use the trained model to predict scores on the test data,
compare with actual scores, and calculate the Mean Absolute Error (MAE).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------
# 1. Recreate the dataset
# ---------------------------------------------------
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}
df = pd.DataFrame(data)

# ---------------------------------------------------
# 2. Features (X) and target (y)
# ---------------------------------------------------
X = df[['Hours']]
y = df['Score']

# ---------------------------------------------------
# 3. Split and train (same as Task 2)
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ---------------------------------------------------
# 4. Predict on test data
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# 5. Calculate error (Mean Absolute Error)
# ---------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f} points")

# ---------------------------------------------------
# 6. See predictions vs actuals
# ---------------------------------------------------
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:")
print(results)

# ---------------------------------------------------
# 7. Predict for a new student who studied 4.5 hours
# ---------------------------------------------------
new_hours = [[4.5]]
predicted_score = model.predict(new_hours)
print(f"\nPredicted score for 4.5 hours: {predicted_score[0]:.2f}")
