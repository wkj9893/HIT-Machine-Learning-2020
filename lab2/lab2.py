import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_data(mu1=0, sigma1=0.2, mu2=1, sigma2=0.4, size=100, proportion=0.5):
    X = np.zeros((size, 3))
    Y = np.zeros((size, 1))
    pos_size = int(size * proportion)
    neg_size = size - pos_size
    x1 = np.random.normal(mu1, sigma1, pos_size)
    x2 = np.random.normal(mu1, sigma1, pos_size)
    x3 = np.random.normal(mu2, sigma2, neg_size)
    x4 = np.random.normal(mu2, sigma2, neg_size)
    X[:, 0] = [1 for i in range(size)]
    X[:, 1] = np.hstack((x1, x3))
    X[:, 2] = np.hstack((x2, x4))
    for i in range(pos_size):
        Y[i] = 1
    for j in range(neg_size):
        Y[pos_size + j] = 0
    return X, Y


def plot_data(X, W, Y, fit=False):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for i in range(len(Y)):
        if Y[i][0] == 1:
            x1.append(X[i][1])
            x2.append(X[i][2])
        else:
            x3.append(X[i][1])
            x4.append(X[i][2])
    plt.scatter(x1, x2, label="positive")
    plt.scatter(x3, x4, label="negative")
    if fit:
        x = [i / 10 for i in range(-10, 30)]
        y = [- (W[0][0] + W[1][0] * i) / W[2][0] for i in x]
        plt.plot(x, y)

    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend()


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# hypothesis = sigmoid(X @ W)
def hypothesis(X, W):
    return sigmoid(X @ W)


# Vectorized cost function     cost = (-y'log(h)-(1-y)'log(1-h))/m
def cost_function(X, W, Y):
    cost = 0
    matrix_one = np.ones((Y.shape[0], Y.shape[1]))
    h = hypothesis(X, W)
    m = len(Y)
    cost += (-Y.T @ np.log(h) - (matrix_one - Y).T @ np.log(matrix_one - h)) / m
    cost = float(cost)
    return cost


def score(X, W, Y):
    score = 0
    size = Y.shape[0]
    h = X @ W
    for i in range(size):
        prediction = 1 if h[i][0] > 0 else 0
        if prediction == Y[i][0]:
            score += 1
    return score / size

def train_test_split(X, Y, test_size=0.2):
    test_num = int(test_size * Y.shape[0])
    train_num = Y.shape[0] - test_num
    train_X = X[:train_num, :-1]
    train_Y = Y[:train_num, -1:]
    test_X = X[train_num:, :-1]
    test_Y = Y[train_num:, -1:]
    return train_X, train_Y, test_X, test_Y


def gradient_descent(X, Y, learning_rate=0.1, iteration_time=0):
    W = np.zeros((X.shape[1], 1))
    m = Y[0]
    cost = cost_function(X, W, Y)
    if iteration_time > 0:
        for i in range(iteration_time):
            z = X @ W
            h = sigmoid(z)
            W = W - (learning_rate / m) * X.T @ (h - Y)
            if cost < cost_function(X, W, Y):
                learning_rate = 1 / 2 * learning_rate
        return W
    while True:
        if cost < 1e-1:
            break
        z = X @ W
        h = sigmoid(z)
        W = W - (learning_rate / m) * X.T @ (h - Y)
        if cost < cost_function(X, W, Y):
            learning_rate = 1 / 2 * learning_rate
    return W


def gradient_descent_with_regulation(X, Y, learning_rate=0.1, Lambda=0.1, iteration_time=0):
    W = np.zeros((X.shape[1], 1))
    m = Y[0]
    cost = cost_function(X, W, Y)
    if iteration_time > 0:
        for i in range(iteration_time):
            z = X @ W
            h = sigmoid(z)
            W = W - (learning_rate / m) * (X.T @ (h - Y) + Lambda / m * W)
            if cost < cost_function(X, W, Y):
                learning_rate = 1 / 2 * learning_rate
        return W
    while True:
        if cost < 1e-1:
            break
        z = X @ W
        h = sigmoid(z)
        W = W - (learning_rate / m) * (X.T @ (h - Y) + Lambda / m * W)
        if cost < cost_function(X, W, Y):
            learning_rate = 1 / 2 * learning_rate
    return W


def newton(X, Y, iteration_time):
    W = np.zeros((X.shape[1], 1))
    n, m = X.shape
    for i in range(iteration_time):
        h = hypothesis(X, W)
        U = 1 / n * X.T @ (h - Y)  # (m,1)
        H = 1 / n * np.dot(np.dot(np.dot(X.T, np.diag(h.reshape(n))), np.diag(1 - h.reshape(n))), X)  # (m,m)
        W = W - np.linalg.inv(H) @ U  # (m,1)
    return W


def newton_with_regulation(X, Y, iteration_time, Lambda):
    W = np.zeros((X.shape[1], 1))
    n, m = X.shape
    for i in range(iteration_time):
        h = hypothesis(X, W)
        U = 1 / n * (X.T @ (h - Y) + Lambda * W)  # (m,1)
        H = 1 / n * np.dot(np.dot(np.dot(X.T, np.diag(h.reshape(n))), np.diag(1 - h.reshape(n))), X)  # (m,m)
        W = W - np.linalg.inv(H) @ U - Lambda / n * W # (m,1)
    return W


X, Y = generate_data()
plot_W = np.ones((X.shape[1], 1))
plot_data(X, plot_W, Y)
plt.show()

# gradient descent without regulation
time = [i for i in range(1, 100)]
cost = []
prediction = []
for iteration_time in range(1, 100):
    W = gradient_descent(X, Y, iteration_time=iteration_time)
    cost.append(cost_function(X, W, Y))
    prediction.append(score(X, W, Y))
# plot cost with iteration time
plt.plot(time, cost)
plt.title('#gradient descent without regulation')
plt.xlabel('iteration time')
plt.ylabel('cost')
plt.show()
# plot score with iteration time
plt.plot(time, prediction)
plt.title('#gradient descent without regulation')
plt.xlabel('iteration time')
plt.ylabel('prediction score')
plt.show()
# fit plot
plot_data(X, W, Y, fit=True)
plt.show()

# gradient descent with regulation
time = [i for i in range(1, 100)]
cost = []
prediction = []
for iteration_time in range(1, 100):
    W = gradient_descent_with_regulation(X, Y, iteration_time=iteration_time)
    cost.append(cost_function(X, W, Y))
    prediction.append(score(X, W, Y))
plt.plot(time, cost)
plt.title('gradient descent with regulation')
plt.xlabel('iteration time')
plt.ylabel('cost')
plt.show()
# plot score with iteration time
plt.plot(time, prediction)
plt.title('#gradient descent with regulation')
plt.xlabel('iteration time')
plt.ylabel('prediction score')
plt.show()
# fit plot
plot_data(X, W, Y, fit=True)
plt.show()


# newton method
time = [i for i in range(1, 10)]
cost = []
prediction = []
for iteration_time in range(1, 10):
    W = newton(X, Y, iteration_time=iteration_time)
    cost.append(cost_function(X, W, Y))
    prediction.append(score(X, W, Y))
plt.plot(time, cost)
plt.title('newton method')
plt.xlabel('iteration time')
plt.ylabel('cost')
plt.show()
# plot score with iteration time
plt.plot(time, prediction)
plt.title('newton method')
plt.xlabel('iteration time')
plt.ylabel('prediction score')
plt.show()
# fit plot
plot_data(X, W, Y, fit=True)
plt.show()

#read data from uci
data = pd.read_csv('data_banknote_authentication.txt')
x = data.iloc[:, :-1]
y = data.iloc[:, -1:]
X = np.array(x)
Y = np.array(y)


# newton method
time = [i for i in range(1, 10)]
cost = []
prediction = []
for iteration_time in range(1, 10):
    W = newton(X, Y, iteration_time=iteration_time)
    cost.append(cost_function(X, W, Y))
    prediction.append(score(X, W, Y))
plt.plot(time, cost)
plt.title('newton method')
plt.xlabel('iteration time')
plt.ylabel('cost')
plt.show()
# plot score with iteration time
plt.plot(time, prediction)
plt.title('newton method')
plt.xlabel('iteration time')
plt.ylabel('prediction score')
plt.show()


train_X, train_Y, test_X, test_Y = train_test_split(X, Y, test_size=0.2)
train_W = newton(train_X, train_Y, iteration_time=10)
print(f'对train data的分类准确率是{score(train_X, train_W, train_Y)}')
print(f'对test data训练的分类准确率是{score(test_X, train_W, test_Y)}')

train_W = newton_with_regulation(train_X, train_Y, iteration_time=10,Lambda=0.1)
print(f'对train data的分类准确率是{score(train_X, train_W, train_Y)}')
print(f'对test data训练的分类准确率是{score(test_X, train_W, test_Y)}')
