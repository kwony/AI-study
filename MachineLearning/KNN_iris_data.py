# # 4. KNN sklearn
# 
# 자기 주변에서 가장 가까운 K개의 이웃의 속성을 보고 군집화하는 알고리즘

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

iris = datasets.load_iris()
x = iris.data[:, 0:2]
y = iris.target

model = neighbors.KNeighborsClassifier(6)
model.fit(x, y)
model.predict([[8.3, 2.4]])

plt.figure(figsize=(10, 5))
plt.scatter(x[:,0], x[:, 1])
plt.title("Data points")

plt.show()

model = neighbors.KNeighborsClassifier(30)
model.fit(x, y)

x_min, x_max = x[:, 0].min() - 1, x[:,0].max() +1
y_min, y_max = x[:, 1].min() - 1, x[:,1].max() +1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#0000FF']) 

plt.figure(figsize=(10,5))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolors='gray')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 1)")
plt.show()
