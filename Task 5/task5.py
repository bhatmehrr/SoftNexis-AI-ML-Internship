"""
Project 2: Flower Classifier
Task 5: Train a k-Nearest Neighbors (k-NN) Classifier

Goal: Build a classifier that predicts a flower's species by looking
at the "k" most similar flowers from the training set and taking a
majority vote.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------
# 1. Load and split the data (same as Task 4)
# ---------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ---------------------------------------------------
# 2. Create k-NN model with k=3 (look at 3 nearest neighbours)
# ---------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=3)

# ---------------------------------------------------
# 3. Train the model
# ---------------------------------------------------
knn.fit(X_train, y_train)

# ---------------------------------------------------
# 4. Predict on test set
# ---------------------------------------------------
y_pred = knn.predict(X_test)

# ---------------------------------------------------
# 5. Check accuracy
# ---------------------------------------------------
accuracy = knn.score(X_test, y_test)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------
# 6. Try a sample prediction
# ---------------------------------------------------
sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
predicted_class = knn.predict(sample_flower)
print(f"\nSample measurements {sample_flower[0]} "
      f"predicted as: {iris.target_names[predicted_class[0]]}")
