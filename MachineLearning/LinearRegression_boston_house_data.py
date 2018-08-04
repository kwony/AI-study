# coding: utf-8
# # 1. LSE boston house data
# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[3]:
df_data = pd.read_excel('boston_house_data.xlsx', encoding='utf-8')
df_data.head()


# In[4]:
df_target = pd.read_excel('boston_house_target.xlsx', encoding='utf-8')
df_target.head()


# In[5]:
df_main = pd.concat([df_data, df_target], axis=1)
df_main.head()


# In[18]:
df_test = df_main.rename(columns={ 1 : 'test'}, inplace=True)
df_main.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'] 
df_main.head()


# In[19]:
df_main.describe()


# In[20]:
boston_data = np.array(df_data)
boston_target = np.array(df_target)


# In[21]:
boston_data.shape


# In[22]:
type(boston_data)


# In[70]:
boston_data.shape


# In[29]:
boston_X = boston_data[:, 12:13]
boston_X


# In[71]:
boston_Y = boston_target
boston_Y


# In[72]:
boston_data.shape
boston_target.shape


# In[74]:
boston_X.shape
boston_Y.shape


# In[78]:
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(boston_X, boston_Y, test_size=0.3, random_state=0)


# In[79]:
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[80]:
from sklearn import linear_model

model = linear_model.LinearRegression()

print(x_train.shape)
print(y_train.shape)


# In[81]:
model.fit(x_train, y_train)


# In[82]:
print('Coefficients: ', model.coef_)

# In[84]:
dir(model)


# In[85]:
model.predict(x_train)


# In[87]:
np.mean(model.predict(x_test) - y_test) **2


# In[89]:
print('MSE(Training data): ', np.mean((model.predict(x_train) - y_train) ** 2))


# In[91]:
from sklearn.metrics import mean_squared_error
print('MSE(Test data) : ', mean_squared_error(model.predict(x_test), y_test))


# In[92]:
np.sqrt(mean_squared_error(model.predict(x_test), y_test))


# In[93]:
plt.figure(figsize=(10,10))

plt.scatter(x_test, y_test, color="black")
plt.scatter(x_train, y_train, color="red", s=1)

plt.plot(x_test, model.predict(x_test), color="blue", linewidth=3)

plt.show()

