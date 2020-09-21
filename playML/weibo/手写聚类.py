from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
data = iris.data

"""
1、随机生成中心
2、求样本到中心的聚类（欧式距离）
3、将样本归类
4、寻找新类中心
5、判定一下，如果算法稳定，跳出，如果不稳定跳回第二部
"""
n = len(data)
k = 3

# 第一步
center = data[:k, :]
center_new = np.zeros([k, data.shape[1]])

# 第二步，拿样本减去中心
dist = np.zeros([n, k + 1])
while True:
    for i in range(n):
        for j in range(k):
            dist[i, j] = np.sqrt(sum((data[i, :] - center[j, :]) ** 2))  # 算出一个元素到中心的距离

        # 第三步
        dist[i, k] = np.argmin(dist[i, :k])

    # 第四步 求他们每一列的均值
    for i in range(k):
        index = dist[:, k] == i
        center_new[i, :] = data[index, :].mean(axis=0)

    # 第五步 判定
    if np.all(center == center_new):  # 想得到全为真
        break
    center = center_new
