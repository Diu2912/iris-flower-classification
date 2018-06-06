import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def class_transform(class_name: str):
    if class_name == "Iris-versicolor":
        return 0
    elif class_name == "Iris-setosa":
        return 1
    elif class_name == "Iris-virginica":
        return 2

def class_transform_back(class_id: int):
    if class_id == 0:
        return "versicolor"
    elif class_id == 1:
        return "setosa"
    elif class_id == 2:
        return "virginica"

data: pd.DataFrame = pd.read_csv("iris_data.csv", names = ["sepal length", "sepal width", "petal length", "petal width", "class"])
data["class"]: pd.Series = data["class"].apply(class_transform)

# split into train and test
train: pd.DataFrame
test: pd.DataFrame
train, test = train_test_split(data, test_size = 0.2)

# represent classes by numbers
"""
train.loc[train["class"] == "versicolor", "class"] = 0
train.loc[train["class"] == "setosa", "class"] = 1
train.loc[train["class"] == "virginica", "class"] = 2
test.loc[train["class"] == "versicolor", "class"] = 0
test.loc[train["class"] == "setosa", "class"] = 1
test.loc[train["class"] == "virginica", "class"] = 2
"""

# training info
X: pd.DataFrame = train.iloc[ : , 0:4] # all rows, only features (sepal and petal width and length)
y: pd.Series = train.iloc[ : , -1] # all rows, only target (class) column, as series
y = pd.to_numeric(y) # change type from object to int64

# test info
y_test: pd.Series = pd.to_numeric(test.iloc[ : , -1]) # target column of test dataframe, as int64


# apply k-nearest neighbors algorithm
KNN = KNeighborsClassifier()
KNN.fit(X, y) # train
predictions = KNN.predict(test.iloc[ : , : -1]) # predict
comparison = pd.DataFrame({"test_data": y_test, "prediction": predictions}) # compare test data and predicted results
comparison["test_data"] = comparison["test_data"].apply(class_transform_back)
comparison["prediction"] = comparison["prediction"].apply(class_transform_back)
incorrect = comparison[comparison.test_data != comparison.prediction]

print(comparison)

print("\nAccuracy score: ", accuracy_score(y_test, predictions))