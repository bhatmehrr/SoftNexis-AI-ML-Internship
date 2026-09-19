# Project 1: Student Performance Predictor

Predict a student's exam score based on hours studied.

## Task 1: Load and Explore the Data

**Goal:** Create/load a dataset containing "Hours Studied" and "Exam Score",
then inspect it to understand its structure before building any model.

### What this script does
- Creates a small sample dataset (Hours vs Score)
- Displays the first 5 rows
- Prints summary statistics (mean, min, max, etc.)
- Checks for missing values
- Confirms data types
- Prints the dataset shape

### How to run

```bash
pip install -r requirements.txt
python task1_load_explore_data.py
```

### Tools
- Python
- pandas

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

---

# Project 2: Flower Classifier

Classify iris flowers into three species (setosa, versicolor,
virginica) using petal and sepal measurements.

## Task 4: Prepare the Data (Split into Train/Test)

**Goal:** Load the Iris dataset, separate features (measurements)
from labels (species), then randomly split into training and
testing sets.

### What this script does
- Loads the built-in Iris dataset from scikit-learn
- Separates features (`X`) from labels (`y`)
- Splits into 70% training / 30% testing
- Prints the number of training and test samples
- Prints the feature names and target class names

### How to run

```bash
pip install -r requirements.txt
python task4_prepare_data.py
```

### Tools
- Python
- scikit-learn (Iris dataset built-in)

### Expected output
105 training samples, 45 test samples (70/30 split of 150 total).

### Next steps
Later tasks will likely train a classifier (e.g. KNN or logistic
regression) on this split and evaluate its accuracy.

## Task 5: Train a k-Nearest Neighbors (k-NN) Classifier

**Goal:** Build a classifier that predicts a flower's species by
looking at the "k" most similar flowers from the training set and
taking a majority vote.

### What this script does
- Loads and splits the Iris dataset (same as Task 4)
- Creates a k-NN model with k=3 (looks at the 3 nearest neighbours)
- Trains the model on the training set
- Predicts species on the test set
- Prints the test accuracy
- Predicts the species for one sample set of measurements

### How to run

```bash
pip install -r requirements.txt
python task5_knn_classifier.py
```

### Tools
- Python
- scikit-learn

### Expected output
Accuracy around 95% or higher — k-NN works very well on the Iris
dataset since the three species are fairly well separated by their
measurements.

### Next steps
Future tasks may compare k-NN against other classifiers (e.g.
logistic regression, decision trees) or tune the value of k.

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
