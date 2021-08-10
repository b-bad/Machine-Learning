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
        《统计计算方法》P56 算法3.3(1)
        找到“当前最近点”
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
            sorted_label = sorted(label_dict.items(), key=lambda item:item[1],reverse=True)  # 给标签排序
            return sorted_label[0][0]

        itemArray = np.array(item)
        node = self.find_nearest_node(item=item)
        if node == None:
            return None
        node_list = []  # 按到目标点距离排序保存的节点列表
        dis = np.sqrt(sum(itemArray - node.data)**2)
        least_dis = dis
        node_list.append([least_dis, tuple(node.data), node.label])

        if node.left_child != None:
            # 由find_nearest_node()获取的当前最近点可能不是叶节点
            # 此时有两种情况
            # 1.itemArray[current_dim] < node.data[current_dim]且left_child == None.
            #   若非叶节点，则应该存在一个right_child 
            #   但由于kdTree的生成规则，当一个分支只剩下两个元素时，仅有“一个元素为父节点，另一个元素为左分支的情况”
            #   故排除情况1
            # 2.itemArray[current_dim] > node.data[current_dim]且right_child == None.
            #   同理，这种情况可以成立，故只需考虑该节点的left_child是否为None的情况
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
            《统计计算方法》P56 算法3.3(3)
            递归向上回退
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
                        self.left_search(itemArray, node_list, k, other_child)  # 目标点在超平面左侧，所以从左侧开始搜索，节省时间成本
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
        sorted_label = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)  # 给标签排序
        return sorted_label[0][0]

    def left_search(self, item, node_list, k, node:Node):

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        if node.left_child == None and node.right_child == None:
            # 叶节点
            dis = np.sqrt(sum((item - node.data)**2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.data), node.label])
            return
        # 先进行左子树搜索 更新node_list
        self.left_search(item, node_list, k, node.left_child)

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # 再比较根节点
        dis = np.sqrt(sum((item - node.data)**2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.data), node.label])
        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # 最后比较右子树
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
            # 叶节点
            dis = np.sqrt(sum((item - node.data)**2))
            if k > len(node_list) or dis < least_dis:
                node_list.append([dis, tuple(node.data), node.label])
            return
        # 先进行右子树搜索 更新node_list
        self.right_search(item, node_list, k, node.right_child)

        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # 再比较根节点
        dis = np.sqrt(sum((item - node.data)**2))
        if k > len(node_list) or dis < least_dis:
            node_list.append([dis, tuple(node.data), node.label])
        node_list.sort()
        if k >= len(node_list):
            least_dis = node_list[-1][0]
        else:
            least_dis = node_list[k-1][0]
        # 最后比较左子树
        if k > len(node_list) or abs(item[node.data_dim] - node[node.data_dim]):
            if node.right_child != None:
                self.right_search(item, node_list, k, node.left_child)
        
