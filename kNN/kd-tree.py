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

    def find_nearest_node(self, item)->Node:
        """
        ????????????????????????P56 ??????3.3(1)
        ???????????????????????????
        """
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
    
    def knn(self, item, k=1):

        if self.length <= k:
            label_dict = {}
            for element in self.transfer_list(self.__root):
                if element['label'] in label_dict:
                    label_dict[element['label']] += 1
                else:
                    label_dict[element['label']] = 1
            sorted_label = sorted(label_dict.items(), key=lambda item:item[1],reverse=True)  # ???????????????
            return sorted_label[0][0]

        itemArray = np.array(item)
        node = self.find_nearest_node(item=item)
        if node == None:
            return None
        node_list = []  # ????????????????????????????????????????????????
        dis = np.sqrt(sum(itemArray - node.data)**2)
        least_dis = dis
        node_list.append([least_dis, tuple(node.data), node.label])

        if node.left_child != None:
            # ???find_nearest_node()?????????????????????????????????????????????
            # ?????????????????????
            # 1.itemArray[current_dim] < node.data[current_dim]???left_child == None.
            #   ???????????????????????????????????????right_child 
            #   ?????????kdTree???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            #   ???????????????1
            # 2.itemArray[current_dim] > node.data[current_dim]???right_child == None.
            #   ???????????????????????????????????????????????????????????????left_child?????????None?????????
            left_child = node.left_child
            left_dis = np.sqrt(sum(itemArray - left_child.data)**2)
            if k > len(node_list) or left_dis < least_dis:
                node_list.append([left_dis, tuple(left_child.data), left_child.label])
                node_list.sort()
                if k >= len(node_list):
                    least_dis = node_list[-1][0]
                else:
                    least_dis = node_list[k-1][0]
        
        while True:
            """
            ????????????????????????P56 ??????3.3(3)
            ??????????????????
            """
            if node == self.__root:
                break
            parent = node.parent
            parent_dis = np.sqrt(sum(itemArray - parent.data)**2)
            if k > len(node_list) or parent_dis < least_dis:
                node_list.append([parent_dis, tuple(parent.data), parent.label])
                node_list.sort()
                if k >= len(node_list):
                    least_dis = node_list[-1][0]
                else:
                    least_dis = node_list[k-1][0]
            
            if k > len(node_list) or abs(itemArray[parent.data_dim] - parent[parent.data_dim]) < least_dis:
                if parent.left_child == node:
                    other_child = parent.right_child
                else:
                    other_child = parent.left_child
                if other_child != None:
                    if itemArray[parent.data_dim] <= parent[parent.data_dim]:
                        self.left_search(itemArray, node_list, k, other_child)  # ??????????????????????????????????????????????????????????????????????????????
                    else:
                        self.right_search(itemArray, node_list, k, other_child)
            
            node = parent

        label_dict = {}
        node_list = node_list[:k]
        for element in node_list:
            if element[2] in label_dict:
                label_dict[element[2]] += 1
            else:
                label_dict[element[2]] = 1
        sorted_label = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)  # ???????????????
        return sorted_label[0][0]

    def left_search(self, item, node_list, k, node:Node):

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        if node.left_child == None and node.right_child == None:
            # ?????????
            dis = np.sqrt(sum((item - node.data)**2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.data), node.label])
            return
        # ???????????????????????? ??????node_list
        self.left_search(item, node_list, k, node.left_child)

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # ??????????????????
        dis = np.sqrt(sum((item - node.data)**2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.data), node.label])
        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # ?????????????????????
        if k > len(node_list) or abs(item[node.data_dim] - node[node.data_dim]):
            if node.right_child != None:
                self.left_search(item, node_list, k, node.right_child)

    def right_search(self, item, node_list, k, node:Node):

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        if node.right_child == None and node.left_child == None:
            # ?????????
            dis = np.sqrt(sum((item - node.data)**2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.data), node.label])
            return
        # ???????????????????????? ??????node_list
        self.right_search(item, node_list, k, node.right_child)

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # ??????????????????
        dis = np.sqrt(sum((item - node.data)**2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.data), node.label])
        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # ?????????????????????
        if k > len(node_list) or abs(item[node.data_dim] - node[node.data_dim]):
            if node.right_child != None:
                self.right_search(item, node_list, k, node.left_child)
        
