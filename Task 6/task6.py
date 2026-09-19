"""
Project 2: Flower Classifier
Task 6: Improve the Model with Feature Scaling and Evaluation

Goal: Scale all features to the same range (0 to 1) and re-train the
k-NN model. Compare performance with the unscaled model using a
classification report and confusion matrix.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------
# 1. Load and split the data (same as Task 4)
# ---------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# ---------------------------------------------------
# 2. Baseline: unscaled k-NN (same as Task 5), for comparison
# ---------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
baseline_accuracy = knn.score(X_test, y_test)
print(f"Unscaled test accuracy: {baseline_accuracy * 100:.2f}%")

# ---------------------------------------------------
# 3. Create scaler and transform training & test data
# ---------------------------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use same scaling as training

# ---------------------------------------------------
# 4. Train new k-NN on scaled data
# ---------------------------------------------------
knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------
y_pred_scaled = knn_scaled.predict(X_test_scaled)
new_accuracy = knn_scaled.score(X_test_scaled, y_test)
print(f"Scaled test accuracy: {new_accuracy * 100:.2f}%")

# ---------------------------------------------------
# 6. Detailed report
# ---------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred_scaled, target_names=iris.target_names))

# ---------------------------------------------------
# 7. Confusion matrix (how many were misclassified)
# ---------------------------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_scaled))
