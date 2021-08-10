from numpy.lib.index_tricks import RClass
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

class Node:
    def __init__(self, data=None, label=None, parent=None, data_dim=None, left_child=None, right_child=None) -> None:
        self.data = data
        self.label = label
        self.parent = parent
        self.data_dim = data_dim
        self.left_child = left_child
        self.right_child = right_child

class kdTree:
    def __init__(self, trainSet:list, testSet:list) -> None:
        self.testSet = testSet
        self.trainSet = trainSet
        self.testData = [a[:-1] for a in testSet]
        self.testLabel = [a[-1] for a in testSet]
        self.trainData = [a[:-1] for a in trainSet]
        self.trainLabel = [a[-1] for a in trainSet]
        self.__length = 0
        self.__root = self.__createTree(self.trainData,self.trainData)

    def __createTree(self, dataList, labelList, parentNode=None):
        dataArray = np.array(dataList)
        dataNum, dataDim = dataArray.shape
        labelArray = np.array(labelList).reshape(dataNum, 1)
        if dataNum == 0:
            return None
        val_list = [np.var(dataArray[:, col]) for col in range(dataDim)]
        max_index = val_list.index(max(val_list))
        max_feat_ind_list = dataArray[:, max_index].argsort()
        mid_data_index = max_feat_ind_list[dataNum // 2]
        if dataNum == 1:
            self.__length += 1
            return Node(data=dataArray[mid_data_index], 
                        label=labelArray[mid_data_index],
                        data_dim=mid_data_index,
                        parent=parentNode)
        node = Node(data=dataArray[mid_data_index],
                    data_dim=mid_data_index,
                    label=labelArray[mid_data_index],
                    parent=parentNode)
        left_tree = dataArray[max_feat_ind_list[:dataNum // 2]]
        left_label = labelArray[max_feat_ind_list[:dataNum // 2]]
        left_child = self.__createTree(left_tree, left_label, node)
        if dataNum == 2:
            right_child = None
        else:
            right_tree = dataArray[max_feat_ind_list[dataNum // 2:]]
            right_label = dataArray[max_feat_ind_list[dataNum // 2:]]
            right_child = self.__createTree(right_tree, right_label, node)
        node.left_child = left_child
        node.right_child = right_child
        self.__length += 1
        return node

    def find_nearest_node(self, item):

        itemArray = np.array(item)
        if self.__length == 0:
            return None
        node = self.__root
        if self.__length == 1:
            return node
        while True:
            current_dim = node.data_dim
            if itemArray[current_dim] == node.data[current_dim]:
                return node
            elif itemArray[current_dim] < node.data[current_dim]:
                if node.left_child == None:
                    return node
                else:
                    node = node.left_child
            else:
                if node.right_child == None:
                    return node
                else:
                    node = node.right_child
    