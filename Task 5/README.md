# Project 2: Flower Classifier

Classify iris flowers into three species (setosa, versicolor,
virginica) using petal and sepal measurements.


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