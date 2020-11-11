import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_data(dimension=2):
    if dimension == 2:
        mean = [1, 1]
        cov = [[1, 0.5], [0.5, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        data = np.zeros((x.shape[0], 2))
        data[:, 0] = x
        data[:, 1] = y
        return data


# 标准化
def standardize(data):
    mean = np.mean(data, axis=0)
    data = data - mean
    return data


def pca(data, n_components=1):
    data = standardize(data)
    # 计算协方差矩阵
    cov = np.cov(data.T)
    # 特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # 特征值从大到小的对应序列号
    index = np.argsort(eigenvalues)[::-1]
    # 选取最大的n个特征值对应的特征向量作为主成分
    principal_component = eigenvectors[:, index[0:n_components]]
    return principal_component


def plot_data(n_components):
    image = Image.open('lena.jpg')
    data = np.array(image)
    w = pca(data, n_components)
    mean = np.mean(data, axis=0)
    pca_data = (data - mean) @ w @ w.T + mean
    plt.imshow(pca_data, cmap='gray')


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# generate data and pca
data = generate_data()
w = pca(data)
mean = np.mean(data, axis=0)
pca_data = (data - mean) @ w @ w.T + mean

# plot raw data and pca data
plt.scatter(data[:, 0], data[:, 1], label='raw data')
plt.scatter(pca_data[:, 0], pca_data[:, 1], label='pca data')
plt.legend()
plt.show()

# compare original image and pca image
image = Image.open('lena.jpg')
data = np.array(image)
plt.imshow(data, cmap='gray')
plt.show()
for i in range(10, 100, 20):
    plot_data(n_components=i)
    plt.title(f'dimension = {i}')
    plt.show()
# compute psnr
x = []
psnr = []
for i in range(10, 700, 20):
    x.append(i)
    w = pca(data, n_components=i)
    mean = np.mean(data, axis=0)
    pca_data = (data - mean) @ w @ w.T + mean
    psnr.append(PSNR(data, pca_data))
plt.plot(x, psnr)
plt.show()
