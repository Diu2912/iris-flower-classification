import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

def species_to_int(species: str) -> int:
    if species == "Iris-versicolor": return 0
    elif species == "Iris-setosa": return 1
    elif species == "Iris-virginica": return 2

def species_to_string(species: int) -> str:
    if species == 0: return "Iris-versicolor"
    elif species == 1: return "Iris-setosa"
    elif species == 2: return "Iris-virginica"

data: pd.DataFrame = pd.read_csv("iris_data.csv", names = ["sepal length", "sepal width", "petal length", "petal width", "species"])

print("DATASET SAMPLE:")
print(data.sample(frac = 0.04).to_string())

data["species"]: pd.Series = data["species"].apply(species_to_int)

# split into train and test
train: pd.DataFrame
test: pd.DataFrame
train, test = train_test_split(data, test_size = 0.2) # 80% training, 20% testing

# training info
X: pd.DataFrame = train.iloc[:, 0:-1] # all rows, only features (sepal and petal width and length)
y: pd.Series = train.iloc[:, -1] # all rows, only target (species) column, as series
y = pd.to_numeric(y) # change type from object to int64

# test info
y_test: pd.Series = pd.to_numeric(test.iloc[:, -1]) # target column of test dataframe, as int64

# apply k-nearest neighbors algorithm
KNN = KNeighborsClassifier()
KNN.fit(X, y) # train
predictions = KNN.predict(test.iloc[:, :-1]) # predict

# print results
print("\nKNN PREDICTION RESULTS:")
comparison = pd.DataFrame({"test_data": y_test, "prediction": predictions}) # compare test data and predicted results
comparison["test_data"] = comparison["test_data"].apply(species_to_string)
comparison["prediction"] = comparison["prediction"].apply(species_to_string)
comparison["correct"] = comparison["test_data"] == comparison["prediction"]
print(comparison.to_string())

print("\nACCURACY SCORE: ", accuracy_score(y_test, predictions))