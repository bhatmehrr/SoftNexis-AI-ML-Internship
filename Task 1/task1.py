"""
Project 1: Student Performance Predictor
Task 1: Load and Explore the Data

Goal: Create/load a dataset of Hours Studied vs Exam Score,
then explore its structure before building any model.
"""

import pandas as pd

# ---------------------------------------------------
# 1. Create a simple dataset
# ---------------------------------------------------
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}
df = pd.DataFrame(data)

# ---------------------------------------------------
# 2. Explore the data
# ---------------------------------------------------
print("First 5 rows:")
print(df.head())

print("\nData statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDataset shape (rows, columns):")
print(df.shape)
