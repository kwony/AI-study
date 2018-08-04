# 3. Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, model_selection, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

df_data = pd.read_excel('boston_house_data.xlsx', encoding='utf-8')

df_target = pd.read_excel('boston_house_target.xlsx', encoding='utf-8')

mean_price = df_target[0].mean()

# Add new collumn name 'Label'
df_target['Label'] = df_target[0].apply(lambda x: 1 if x > mean_price else 0)

boston_data = np.array(df_data)
boston_target = np.array(df_target['Label'])

boston_X = boston_data[:, (5, 12)]
boston_Y = boston_target

from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(boston_X, boston_Y, test_size=0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Predict data with test data sets
pred_test = model.predict_log_proba(x_test)
pred_test

print('Accuracy: ', accuracy_score(model.predict(x_test), y_test))

# Display data with auc curve
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=pred_test[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("ROC curve")

plt.show()
