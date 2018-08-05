# # 5. K-means sklearn
# 
# 1. k개의 중심값을 임의로 배정한다.
# 2. 각 데이터마다 중심값까지의 거리를 계산하고 가장 가까운 중심값의 클러스터에 할당.
# 3. 클러스에서 속한 데이터의 평균값으로 중심값을 이동
# 4. 데이터에 대한 클러스터 할당이 변하지 않을 때까지 반복

from sklearn import cluster
from sklearn import datasets
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data[:, 0:2]

kmeans = cluster.KMeans(n_clusters=3, random_state=0).fit(X)
print("Clusters: ", kmeans.labels_)

mean_squared_error(kmeans.labels_, iris.target)

X = iris.data
Y = iris.target

print(X)
print()
print(Y)

estimator = [('k=8', cluster.KMeans(n_clusters=8)),
            ('k=3', cluster.KMeans(n_clusters=3)),
            ('k=3(r)', cluster.KMeans(n_clusters=3, n_init=1, init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

for name, est in estimator:
    fig = plt.figure(fignum, figsize=(7,7))
    ax = Axes3D(fig, elev=48, azim=134)
    est.fit(X)
    labels = est.labels_
    
    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='w', s=100)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12 # 값이 커지면 전체 plot 이 작아짐
    fignum = fignum + 1
    
plt.show()
    
# 최적의 cluster 개수를 찾는 알고리즘.
# 그래프상에서 꺽이는 점을 최적의 cluster의 개수로 본다.
def elbow(X):
    total_distance = []
    for i in range(1, 11):
        model = cluster.KMeans(n_clusters=i, random_state = 0)
        model.fit(X)
        
        total_distance.append(model.inertia_)
    
    plt.plot(range(1, 11), total_distance, marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Total distance (SSE)')
    plt.show()
    
elbow(X)
