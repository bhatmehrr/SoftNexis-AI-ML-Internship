"""
Project 2: Flower Classifier
Task 4: Prepare the Data (Split into Train/Test)

Goal: Load the Iris dataset, separate features from labels, and
split into training and testing sets.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------
iris = load_iris()
X = iris.data      # Features: sepal length, sepal width, petal length, petal width
y = iris.target     # Labels: 0=setosa, 1=versicolor, 2=virginica

# ---------------------------------------------------
# 2. Split: 70% for training, 30% for testing
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# ---------------------------------------------------
# 3. Confirm the split
# ---------------------------------------------------
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print(f"\nFeature names: {iris.feature_names}")
print(f"Target classes: {iris.target_names}")
