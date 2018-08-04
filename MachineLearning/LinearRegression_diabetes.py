
# coding: utf-8

# # 2. LSE Diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
dir(diabetes)

df = pd.DataFrame(diabetes.data)

diabetes_X = diabetes.data[:, 2:3]
diabetes_X.shape

diabetes_Y = diabetes.target
diabetes_X.shape

from sklearn import model_selection

# Split datasets into train and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(diabetes_X, diabetes_Y, test_size=0.3, random_state=0)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

np.sqrt(mean_squared_error(model.predict(x_train), y_train))

# Display prediction data with graph
plt.figure(figsize=(10, 10))
plt.scatter(x_test, y_test, color="black")
plt.scatter(x_train, y_train, color="red", s=1)
plt.plot(x_test, model.predict(x_test), color="blue", linewidth=3)
plt.show()

