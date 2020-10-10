import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class lab1:
    def __init__(self, size, m):
        self.size = size  # 样本数据量
        self.m = m  # 拟合多项式函数的阶数

    def generate_data(self):
        """
        产生size个数据
        return:
        x:x轴点的array
        y:y轴点的array，y=sin(2*pi*x)
        """
        x = np.linspace(start=0, stop=1, num=self.size)
        mu, sigma = 0, 0.1  # mean and standard deviation
        y = np.sin(2 * np.pi * x) + np.random.normal(mu, sigma, size=self.size)
        return x, y

    def trans(self, x, y):
        """
        将x,y两个array根据多项式函数的阶m转化成相应的array X，T
        param x: x轴点的array
        param y: y轴点的array，y=sin(2*pi*x)
        return:
        X，T:对应的两个array
        """
        X = np.zeros((self.size, self.m + 1))
        for i in range(self.m + 1):
            for j in range(self.size):
                X[j, i] = np.power(x[j], i)
        T = y.reshape(self.size, 1)
        return X, T

    def error(self, X, W, T):
        """
        根据X,W,T三个矩阵根据公式计算误差
        param X: 二维array X
        param W: array W
        param T: array T
        return: 误差函数的计算结果
        """
        error = 1 / 2 * (X @ W - T).T @ (X @ W - T)  # error.shape = (1,1)
        return error[0][0]

    def E_RMS(self, error):
        """
        计算误差的方均根值
        return: 误差的的方均根值E_RMS
        """
        return np.sqrt(2 / self.size * error)

    def fit_without_regulation(self, X, T):
        """
        计算未加入正则项的数值解W
        param X: 二维array X
        param T: array T
        return: 模型参数W
        """
        return np.linalg.pinv(X) @ T

    def fit_with_regulation(self, X, T, Lambda=1e-7):
        """
        计算加入正则项的数值解W
        param X: 二维array X
        param T: array T
        param Lambda:权重参数
        return: 模型参数W
        """
        return np.linalg.solve(X.T @ X + Lambda * np.identity(self.m + 1), X.T @ T)

    def gradient_descent(self, X, T, rate=0.1, precision=1e-1, times=0,Lambda=0):
        """
        通过梯度下降法计算给定learning rate和精度要求所需要的迭代次数
        param X: 二维array X
        param T: array T
        param rate: learning rate
        param precision:精度要求，默认为0.1
        param times: 迭代次数
        param Lambda:惩罚项参数，默认为0，即为无正则项的梯度下降
        return: 计算出要求精度所需要的迭代次数和W
        """
        W = np.zeros((self.m + 1, 1))  # 初始化W为全为0的列向量
        # 计算给定迭代次数得到的W
        if times != 0:
            for i in range(times):
                last_error = self.error(X, W, T)
                W = W - rate * (X.T @ (X @ W - T) + Lambda * W)     #Lambda=0时为无正则项的梯度下降
                if self.error(X, W, T) > last_error:                #Lambda>0时为有正则项的梯度下降
                    rate = rate / 2
            return W
        times = 0
        while self.error(X, W, T) > precision:
            last_error = self.error(X, W, T)
            W = W - rate * (X.T @ (X @ W - T) + Lambda * W)
            if self.error(X, W, T) > last_error:
                rate = rate / 2
            times += 1
        return times, W

    def conjugate_gradient(self, X, T, precision=1e-5, times=0,Lambda=0):
        """
        通过共轭梯度法计算给定精度要求所需要的迭代次数,
        参考 http://en.wikipedia.org/wiki/Conjugate_gradient_method

        param X: 二维array X
        param T: array T
        param precision:精度要求，默认为0.1
        param times: 迭代次数
        param Lambda:惩罚项参数，默认为0，即为无正则项的共轭梯度
        return: 要求精度要求所需要的迭代次数和W
        """

        # 转化成Ax = b的形式求解
        A = X.T @ X + Lambda * np.identity(self.m+1)        #Lambda=0时为无正则项的共轭梯度
        x = np.zeros((self.m + 1, 1))                       #Lambda>0时为有正则项的共轭梯度
        b = X.T @ T

        r = b - np.dot(A, x)
        p = r
        rsold = r.T @ r
        # 计算给定迭代次数得到的W
        if times != 0:
            for i in range(times):
                Ap = A @ p
                alpha = rsold[0][0] / np.dot(p.T, Ap)[0][0]
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = r.T @ r
                p = r + (rsnew / rsold) * p
                rsold = rsnew
            return x
        times = 0
        while self.error(X, x, T) > precision:
            Ap = A @ p
            alpha = rsold[0][0] / np.dot(p.T, Ap)[0][0]
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.T @ r
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            times += 1
        return times, x

    def poly(self, X, W):
        """
        通过自变量X和参数W生成相应的多项式函数，方便绘图比较
        param W:自变量值
        return:对应的多项函数值
        """
        result = 0
        for i in range(self.m + 1):
            result += W[i][0] * np.power(X, i)
        return result



# start test
size = 10  # 样本数据量
m = 5  # 拟合多项式函数的阶数
model = lab1(size, m)

# generate train data
x, y = model.generate_data()
# generate test data
x_test, y_test = model.generate_data()
# generate data of size 100
x_100, y_100 = lab1(size=100, m=5).generate_data()
# generate data of size 15
x_15, y_15 = lab1(size=15, m=5).generate_data()


# plot generate data and y = sin(2*pi*x) for comparison
plt.scatter(x, y)
x1 = np.linspace(0, 1, 1000)
plt.plot(x1, np.sin(2 * np.pi * x1))
plt.title('sample data')
plt.xlabel('x')
plt.ylabel('t')
plt.show()

# trans to generate X and T
X, T = model.trans(x, y)
# trans to generate X_Test and T_Test
X_Test, T_Test = model.trans(x_test, y_test)
# Analytical Solution
W_without_regulation = model.fit_without_regulation(X, T)
W_with_regulation = model.fit_with_regulation(X, T, 1e-7)

# 阶数m取不同值时，最佳拟合曲线情况
Train_E_RMS = []
Test_E_RMS = []
for m in range(size):
    train = lab1(size, m)
    train_X, train_T = train.trans(x, y)
    test_X, test_T = train.trans(x_test, y_test)
    fit_without_regulation = model.fit_without_regulation(train_X, train_T)
    if m == 0 or m == 1 or m == 3 or m == 9:
        plt.scatter(x, y)
        plt.plot(x1, np.sin(2 * np.pi * x1))
        plt.plot(x1, train.poly(x1, fit_without_regulation))
        plt.title(f'fit with m = {m}')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()
    train_error = model.error(train_X, fit_without_regulation, train_T)
    test_error = model.error(test_X, fit_without_regulation, test_T)
    Train_E_RMS.append(model.E_RMS(train_error))
    Test_E_RMS.append(model.E_RMS(test_error))

# plot E_RMS with M in training data and test data
M = np.linspace(0, size - 1, size)
plt.plot(M, Train_E_RMS, label='Training')
plt.plot(M, Test_E_RMS, label='Test')
plt.xlabel('M')
plt.ylabel('$E_{RMS}$')
plt.legend()
plt.show()

# compare fit with different size(15 and 100)
train = lab1(size=100, m=9)
plt.scatter(x_100, y_100)
plt.plot(x1, np.sin(2 * np.pi * x1))
train_X, train_T = train.trans(x_100, y_100)
fit_without_regulation = model.fit_without_regulation(train_X, train_T)
plt.plot(x1, train.poly(x1, fit_without_regulation))
plt.title('N=100')
plt.xlabel('x')
plt.ylabel('t')
plt.show()

train = lab1(size=15, m=9)
plt.scatter(x_15, y_15)
plt.plot(x1, np.sin(2 * np.pi * x1))
train_X, train_T = train.trans(x_15, y_15)
fit_without_regulation = model.fit_without_regulation(train_X, train_T)
plt.plot(x1, train.poly(x1, fit_without_regulation))
plt.title('N=15')
plt.xlabel('x')
plt.ylabel('t')
plt.show()

# 分为训练数据和测试数据来对m=5,size=10时plotE_RMS随lambda的变化情况，结果发现lambda=10^-7时效果最好
Train_E_RMS = []
Test_E_RMS = []
M = []
for log10_lambda in range(-10, 0):
    M.append(log10_lambda)
    Lambda = 10 ** log10_lambda
    train_error = model.error(X, model.fit_with_regulation(X, T, Lambda), T)
    test_error = model.error(X_Test, model.fit_with_regulation(X_Test, T_Test, Lambda), T_Test)
    Train_E_RMS.append(model.E_RMS(train_error))
    Test_E_RMS.append(model.E_RMS(test_error))
plt.plot(M, Train_E_RMS, label='Training')
plt.plot(M, Test_E_RMS, label='Test')
plt.xlabel('$log_{10}\lambda$')
plt.ylabel('$E_{RMS}$')
plt.legend()
plt.show()

# size=10,m分别等于0，5，9时的曲线拟合图(有惩罚项与无惩罚项对比)
for size in [10]:
    for m in [0, 5, 9]:
        train = lab1(size, m)
        x, y = train.generate_data()
        train_X, train_T = train.trans(x, y)
        fit_without_regulation = train.fit_without_regulation(train_X, train_T)
        fit_with_regulation = train.fit_with_regulation(train_X, train_T)
        plt.scatter(x, y)
        plt.plot(x1, np.sin(2 * np.pi * x1))
        plt.plot(x1, train.poly(x1, fit_without_regulation))
        plt.title(f'fit without regulation size = {size} m = {m}')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()
        plt.scatter(x, y)
        plt.plot(x1, np.sin(2 * np.pi * x1))
        plt.plot(x1, train.poly(x1, fit_with_regulation))
        plt.title(f'fit with regulation size = {size} m = {m}')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()


# 使用梯度下降法和共轭梯度法达到相同精度所需次数的比较
times, W = model.gradient_descent(X, T, precision=1e-1)
print(f'使用梯度下降法，精度要求在1*10^-1所需次数:{times}')
times, W = model.gradient_descent(X, T, precision=1e-1,Lambda=1e-7)
print(f'使用共轭梯度法，精度要求在1*10^-1所需次数:{times}')
times, W = model.conjugate_gradient(X, T, precision=1e-5)
print(f'使用共轭梯度法，精度要求在1*10^-5所需次数:{times}')

# 相同迭代次数下，比较梯度下降法和共轭梯度法对曲线的拟合情况(设定迭代次数为100)
for size in [10, 30]:
    for m in [3, 9]:
        train = lab1(size, m)
        x, y = train.generate_data()
        train_X, train_T = train.trans(x, y)
        fit_gradient_descent = train.gradient_descent(train_X, train_T, times=100)
        fit_conjugate_gradient = train.conjugate_gradient(train_X, train_T, times=100)
        plt.scatter(x, y)
        plt.plot(x1, np.sin(2 * np.pi * x1))
        plt.plot(x1, train.poly(x1, fit_gradient_descent))
        plt.title(f'fit with gradient descent size = {size} m = {m}')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()
        plt.scatter(x, y)
        plt.plot(x1, np.sin(2 * np.pi * x1))
        plt.plot(x1, train.poly(x1, fit_conjugate_gradient))
        plt.title(f'fit with conjugate gradient size = {size} m = {m}')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

