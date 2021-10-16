import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import random

def load_data(class_1=0, class_2=1):

    dataList, testSet, trainSet = [], [], []
    iris = datasets.load_iris()
    datas = iris['data']
    targets = iris['target']
    for data, target in zip(datas, targets):
        if target in [class_1, class_2]:
            dataList.append(np.append(data, target).tolist())
    random.shuffle(dataList)
    dataSize = len(dataList)
    testSet = dataList[:int(dataSize*0.3)]
    trainSet = dataList[int(dataSize*0.3):]
    for data in testSet:
        if data[-1] == class_1:
            data[-1] = -1
        elif data[-1] == class_2:
            data[-1] = 1
    
    for data in trainSet:
        if data[-1] == class_1:
            data[-1] = -1
        elif data[-1] == class_2:
            data[-1] = 1
        
    
    return testSet, trainSet


class LogisticRegression:

    def __init__(self, X, y, X_test, y_test, lr, epoch) -> None:
        
        self.y = np.reshape(y,(1, -1))
        self.y_test = np.reshape(y_test,(1, -1))
        m, n = X.shape
        self.X = np.column_stack((X, np.ones(m)))
        self.X_dim = n
        self.X_num = m
        self.X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))
        self.W = np.random.normal(0, 0.01, (1, self.X_dim + 1))
        self.lr = lr
        self.epoch = epoch

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def loss(self, y_hat:np.ndarray):
        """
            use NNL(Negative Log Likelihood) as loss func
            input(y_hat) and class member(y) are both column vector
        """
        return -np.log(self.sigmoid(np.diag(y_hat.dot(self.y.T)))).sum()

    def gd(self):
        for i in range(self.X_num):
            self.W += self.sigmoid(-self.y[0, i] * self.W * self.X[i]) * self.y[0, i] * self.X[i] * self.lr

    def train(self):
        l_list = []
        e_cnt = 0
        for i in range(self.epoch):
            l = self.loss(np.dot(self.W, self.X.T))
            self.gd()
            l_list.append(l)
            print("epoch %d : loss is %f" %(i, l))
            e_cnt += 1
            if l < 1e-6:
                break
        plt.title("Loss")
        plt.plot(np.arange(0, e_cnt), l_list)
        plt.show()
    
    def test(self):
        y_hat = self.sigmoid(self.W.dot(self.X_test.T))
        test_size = y_hat.shape[1]
        cnt = 0
        for i in range(test_size):
            if (y_hat[0, i] >= 0.5 and self.y_test[0, i] == 1) or (y_hat[0, i] < 0.5 and self.y_test[0, i] == -1):
                cnt += 1
        print("test acc:%.3f" %(cnt/test_size))




if __name__ == "__main__":

    test_set, train_set = load_data()
    y = np.array(train_set)[:, -1]
    X = np.array(train_set)[:, 0:-1]
    y_test = np.array(test_set)[:, -1]
    X_test = np.array(test_set)[:, 0:-1]
    LR = LogisticRegression(X, y, X_test, y_test, 0.0001, 100)
    LR.train()
    LR.test()
