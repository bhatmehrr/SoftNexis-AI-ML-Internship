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