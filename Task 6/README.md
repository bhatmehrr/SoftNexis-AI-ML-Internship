# Project 2: Flower Classifier

Classify iris flowers into three species (setosa, versicolor,
virginica) using petal and sepal measurements.


## Task 6: Improve the Model with Feature Scaling and Evaluation

**Goal:** Scale all features to the same range (0 to 1) using
MinMaxScaler, re-train the k-NN model, and compare its performance
against the unscaled model with a classification report and
confusion matrix.

### What this script does
- Loads and splits the Iris dataset (same as Task 4)
- Trains a baseline (unscaled) k-NN model for comparison
- Scales features with `MinMaxScaler` (fit on training data only,
  then applied to test data to avoid data leakage)
- Trains a new k-NN model on the scaled data
- Prints the scaled test accuracy
- Prints a full classification report (precision, recall, F1 per class)
- Prints a confusion matrix showing exactly which flowers were
  misclassified

### How to run

```bash
pip install -r requirements.txt
python task6_feature_scaling.py
```

### Tools
- Python
- scikit-learn

### Expected output
Accuracy stays around 95–100% either way, since the Iris dataset's
features are already on fairly similar scales (all measured in cm).
Scaling matters more on datasets where features vary widely in
range — but it's good practice regardless.

### Next steps
Future tasks may explore other algorithms (e.g. decision trees,
logistic regression) or tune hyperparameters like k using cross-validation.
