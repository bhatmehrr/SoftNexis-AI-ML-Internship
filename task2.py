"""
Project 1: Student Performance Predictor
Task 2: Train a Linear Regression Model

Goal: Split the data into training and testing parts, then train a
linear regression model to map hours studied -> exam score.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
X = df[['Hours']]   # Double brackets -> keeps X as a DataFrame (2D), required by sklearn
y = df['Score']     # Single brackets -> Series (1D) is fine for the target

# ---------------------------------------------------
# 3. Split: 80% for training, 20% for testing
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 4. Create and train the model
# ---------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------
# 5. Inspect the learned line
# ---------------------------------------------------
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ---------------------------------------------------
# 6. Try a sample prediction
# ---------------------------------------------------
sample_hours = 6.5
predicted_score = model.predict([[sample_hours]])[0]
print(f"\nPredicted score for {sample_hours} hours studied: {predicted_score:.2f}")
