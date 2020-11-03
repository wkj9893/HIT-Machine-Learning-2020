import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd


def generate_data():
    mean1 = [1, 1]
    cov1 = [[1, 0], [0, 1]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
    mean2 = [3, 5]
    cov2 = [[1, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
    mean3 = [5, 1]
    cov3 = [[1, 0], [0, 1]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 100).T
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.scatter(x3, y3)
    plt.show()

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    data = np.zeros((len(x), 2))
    data[:, 0] = x
    data[:, 1] = y
    return data


# 根据本实验生成数据，求出分类的准确率
def cluster_score(clusters, length=100):
    score = 0
    for k in range(3):
        zero_count = 0
        one_count = 0
        two_count = 0
        for i in range(length):
            if clusters[i] == 0:
                zero_count += 1
            if clusters[i] == 1:
                one_count += 1
            if clusters[i] == 2:
                two_count += 1
        sum = zero_count + one_count + two_count
        if zero_count == max(zero_count, one_count, two_count):
            score += zero_count / sum
        if one_count == max(zero_count, one_count, two_count):
            score += one_count / sum
        if two_count == max(zero_count, one_count, two_count):
            score += two_count / sum
    return score / 3


def get_centers(data, k):
    dimension = data.shape[1]
    mean = np.mean(data, axis=0).reshape((1, dimension))
    std = np.std(data, axis=0).reshape((1, dimension))
    centers = mean + std * np.random.randn(k, dimension)  # (k,dimension)
    return centers


def k_means(data, k=3, iteration_times=1000):
    size = data.shape[0]
    centers = get_centers(data, k)
    distances = np.zeros((size, k))
    pre_clusters = np.zeros((size, 1))
    for i in range(iteration_times):
        # 计算每个样本离每个中心点的距离
        for j in range(k):
            distances[:, j] = np.linalg.norm(data - centers[j], axis=1)
        # 样本对应的类别为距离最近的中心点
        clusters = np.argmin(distances, axis=1)
        # 如果两次簇划分结果相同，停止迭代，返回结果
        if np.array_equal(clusters, pre_clusters):
            return clusters
        # 更新每个类别的中心点
        for j in range(k):
            centers[j] = np.mean(data[clusters == j], axis=0)
        pre_clusters = np.copy(clusters)
    return clusters


def GMM(data, K=3, iteration_times=1000):
    size, dimension = data.shape
    # 初始化均值向量
    mu = get_centers(data, K)
    # 初始化协方差矩阵
    cov = np.zeros((K, dimension, dimension))
    for i in range(K):
        cov[i, :, :] = np.identity(dimension) / 10
    # 初始化混合系数
    alpha = np.ones(size) / size
    # 初始化后验概率矩阵
    gamma = np.zeros((size, K))
    for i in range(iteration_times):
        for k in range(K):
            # Expectation step
            # 计算后验概率矩阵
            gamma[:, k] = alpha[k] * multivariate_normal(mu[k], cov[k]).pdf(data)
        sum = np.sum(gamma, axis=1).reshape(-1, 1)
        gamma /= sum

        # Maximization step
        for k in range(K):
            gamma_k = np.sum(gamma[:, k], axis=0)
            # 计算新均值向量
            mu[k] = np.sum(gamma[:, k].reshape(-1, 1) * data, axis=0) / gamma_k
            # 计算新协方差矩阵
            cov[k] = (gamma[:, k].reshape(-1, 1) * (data - mu[k])).T @ (data - mu[k]) / gamma_k
            # 计算新混合系数
            alpha[k] = gamma_k / K
    return mu, cov, alpha


def GMM_predict(data, K=3, iteration_times=1000):
    size, dimension = data.shape
    mu, cov, alpha = GMM(data, K, iteration_times)
    gamma = np.zeros((size, K))
    for k in range(K):
        # Expectation step
        # 计算后验概率矩阵
        gamma[:, k] = alpha[k] * multivariate_normal(mu[k], cov[k]).pdf(data)
    sum = np.sum(gamma, axis=1).reshape(-1, 1)
    gamma /= sum
    return np.argmax(gamma, axis=1)


if __name__ == '__main__':
    data = generate_data()
    clusters = k_means(data)
    print(f'通过k-means算法对生成数据的分类准确率为{cluster_score(clusters)}')
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.show()
    clusters = GMM_predict(data)
    print(f'通过GMM算法对生成数据的分类准确率为{cluster_score(clusters)}')
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.show()

    # read data from uci
    data = pd.read_csv('iris.data', header=None).iloc[:, :-1]
    data = pd.DataFrame(data, dtype=float)
    data = np.array(data)

    clusters = k_means(data)
    print(f'通过k-means算法对uci数据的分类准确率为{cluster_score(clusters,length=50)}')
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.show()

    clusters = GMM_predict(data)
    print(f'通过GMM算法对生成数据的分类准确率为{cluster_score(clusters,length=50)}')
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.show()