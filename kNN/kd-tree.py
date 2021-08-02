from sklearn import datasets
import numpy as np
import random


def load_data():

    dataSet, testSet, trainSet = [], [], []
    iris = datasets.load_iris()
    datas = iris['data']
    targets = iris['target']
    for data, target in zip(datas, targets):
        dataSet.append(np.append(data, target).tolist())
    random.shuffle(dataSet)
    dataSize = len(dataSet)
    testSet = dataSet[:int(dataSize*0.3)]
    trainSet = dataSet[int(dataSize*0.3):]
    
    return testSet, trainSet

class kdTree:
    def __init__(self, trainSet:list, testSet:list) -> None:
        self.trainSet = trainSet
        self.testSet = testSet