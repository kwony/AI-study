# # 6. PCA sklearn
# 
# 차원 축소를 통해 최소 차원의 정보로 원래 차원의 정보를 모사하는 알고리즘

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

model = decomposition.PCA(n_components=1)
model.fit(x)
x1 = model.transform(x)

print(x.shape)
print(x1.shape)

sns.distplot(x1[y==0], color="b", bins=20, kde=False)
sns.distplot(x1[y==1], color="g", bins=20, kde=False)
sns.distplot(x1[y==2], color="r", bins=20, kde=False)

plt.xlim(-6, 6)
plt.show()

model = decomposition.PCA(n_components=3)
model.fit(x)
x = model.transform(x)

print(x.shape)

plt.scatter(x[:, 0], x[:, 1], c=iris.target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, elev=48, azim=134) # Set the elevation and azimuth of the axes. (축의 고도와 방위각)

ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=iris.target, edgecolor='w', s=100)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.dist = 12 # 값이 커지면 전체 plot 이 작아짐

plt.show()

