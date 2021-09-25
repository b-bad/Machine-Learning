import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import random
import argparse
import seaborn


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
            data[-1] = 0
        elif data[-1] == class_2:
            data[-1] = 1
    
    for data in trainSet:
        if data[-1] == class_1:
            data[-1] = 0
        elif data[-1] == class_2:
            data[-1] = 1
        
    
    return testSet, trainSet

class leastSquare:

    def __init__(self, testSet:list, trainSet:list, learningRate=0.001, threshold=0.00001, epochs=50, useAdam=False) -> None:
        
        self.testSet = testSet
        self.trainSet = trainSet
        self.testData = [a[:-1] for a in testSet]
        self.testTarget = [a[-1] for a in testSet]
        self.trainData = [a[:-1] for a in trainSet]
        self.trainTarget = [a[-1] for a in trainSet]
        self.learningRate = learningRate
        self.threshold = threshold
        self.X_dim = len(self.testSet[0]) - 1
        self.testSize = len(self.testSet)
        self.trainSize = len(self.trainSet)
        self.beta = np.ones(self.X_dim + 1)
        self.error = 1e6
        self.epochs = epochs
        self.X = np.c_[np.ones(self.trainSize), np.array(self.trainData)]
        self.Y = np.array(self.trainTarget)
        self.trainedFlag = False
        self.pre_labels = []
        self.pre_y = []
        self.useAdam = useAdam

    def calSquareError(self, B):

        Y = self.Y
        X = self.X
        error = 1 / self.trainSize * np.matmul((Y - np.matmul(X, B)).T, (Y - np.matmul(X, B)))

        return error

    def gradientDescent(self):

        Y = self.Y
        X = self.X
        for i in range(self.epochs):
            B = self.beta
            betaTemp = self.beta
            grad = 2 / self.trainSize * (-np.matmul(X.T, Y) + np.matmul(np.matmul(X.T, X), B))
            self.beta -= self.learningRate * grad
            if i % 100 == 0:
                print("epoch %d : error is %f" %(i, self.calSquareError(self.beta)))
            '''if pow(self.calSquareError(betaTemp) - self.calSquareError(self.beta), 2) <= self.threshold:
                print("Finish after epoch %d !!! beta=" %i, self.beta.tolist()[1:])
                break'''
        print("Finish after epoch %d !!! beta=" %i, self.beta.tolist()[:])
        self.trainedFlag = True
        return 0
    
    def predict(self, train=False, threshold=0.5):

        self.pre_labels = []
        if self.trainedFlag == False:
            print("Haven't been trained")
            return 0
        if train:
            data = self.trainData
            target = self.trainTarget
        else:
            data = self.testData
            target = self.testTarget
        truePositive, falsePositive, trueNegative, falseNegative = 0, 0, 0, 0
        beta = self.beta.tolist()
        for x, y in zip(data, target):
            # print(x, beta[1:])
            y_pre = beta[0] + float(np.matmul(np.array(x), np.array(beta[1:]).T))
            self.pre_y.append(round(y_pre, 2))
            # print("target: %d  predict: %f" %(int(y), y_pre))
            if y == 1 and y_pre >= threshold:
                truePositive += 1
                self.pre_labels.append(1)
            elif y == 1 and y_pre < threshold:
                falseNegative += 1
                self.pre_labels.append(0)
            elif y == 0 and y_pre >= threshold:
                falsePositive += 1
                self.pre_labels.append(1)
            else:
                trueNegative += 1
                self.pre_labels.append(0)
        ACC = (truePositive + trueNegative) / len(data)
        PPV = truePositive / (truePositive + falsePositive)
        TPR = truePositive / (truePositive + falseNegative)
        TNR = trueNegative / (trueNegative + falseNegative)
        F1_score = 2 * PPV * TPR / (PPV + TPR)
        print("ACC: %f" %ACC)
        print("PPV: %f" %PPV)
        print("Recall: %f" %TPR)
        print("TNR: %f" %TNR)
        print("F1-score: %f" %F1_score)
        
        return 0
        
    def show_confusion_matrix(self, train=False):

        if train:
            cm = confusion_matrix(self.trainTarget, self.pre_labels, labels=[0, 1])
        else:
            cm = confusion_matrix(self.testTarget, self.pre_labels, labels=[0, 1])
        seaborn.set()
        fig, ax = plt.subplots()
        seaborn.heatmap(cm, annot=True, ax=ax)
        ax.set_title("confusion matrix")
        ax.set_xlabel("predict")
        ax.set_ylabel("true")
        plt.show()

    def show_ROC(self, train=False):

        target = self.testTarget
        if train:
            target = self.trainTarget
        FPR, TPR, threshold = roc_curve(target, self.pre_y, pos_label=1)
        AUC = auc(FPR, TPR)

        plt.figure()
        plt.title("ROC CURVE (AUC = %f)" %AUC)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.plot(FPR,TPR,color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("t_1", type=int, default=0)
    parser.add_argument("t_2", type=int, default=1)
    args = parser.parse_args()
    print("Fitting target %d and %d" %(args.t_1, args.t_2))
    testSet, trainSet = load_data(args.t_1, args.t_2)
    classifier = leastSquare(testSet, trainSet)
    classifier.gradientDescent()
    #classifier.predict(train=True)
    classifier.predict()
    classifier.show_confusion_matrix()
    classifier.show_ROC()