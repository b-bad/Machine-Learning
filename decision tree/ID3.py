from math import log
import collections
# test

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    charas = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, charas

def calLabelProbability(dataSet:list):
    '''计算一个子集中各label的概率'''
    data_size = len(dataSet)
    labels = [data[-1] for data in dataSet]
    count = {}
    P = {}
    for label in labels:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1
    for label in count:
        P[label] = float(count[label] / data_size)
    return P

def calCharaProbability(dataSet:list, index:int):
    '''计算一个子集中根据某chara的概率'''
    data_size = len(dataSet)
    count = collections.Counter([data[index] for data in dataSet])
    
    P = {}
    for key in count:
        P[key] = float(count[key] / data_size)
    
    return P

def calEntropy(dataSet:list):
    '''计算H, H=-Σpi*logpi'''
    P = calLabelProbability(dataSet)
    entropy = 0
    
    for label in P:
        entropy -= P[label] *  log(P[label], 2)

    return entropy

def calInformationGain(dataSet:list, index:int):
    '''计算信息增益g=H(D)-H(D|A)'''
    empirical_entropy = calEntropy(dataSet=dataSet)
    empirical_conditional_entropy = 0
    P = calCharaProbability(dataSet, index)
    for key in P:
        childSet = []
        for data in dataSet:
            if data[index] == key:
                childSet.append(data)
        empirical_conditional_entropy += P[key] * calEntropy(childSet)
    return empirical_entropy - empirical_conditional_entropy

def chooseBestChara(dataSet:list, charas:list):

    charas_size = len(charas)
    G = []
    for i in range(charas_size):
        G.append(calInformationGain(dataSet, i))
        print("chara " + str(i) + " " + str(calInformationGain(dataSet, i)))
    best_chara_index = G.index(max(G))
    return best_chara_index

def splitDataSet(dataSet:list, index:int, value:int):

    retDataSet = []                                     
    for featVec in dataSet:                             
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]             
            reducedFeatVec.extend(featVec[index+1:])     
            retDataSet.append(reducedFeatVec)
    return retDataSet

def generateTree(dataSet:list, charas:list, bestCharas:list):

    data_size = len(dataSet)
    labels = [data[-1] for data in dataSet]
    labels_count = collections.Counter(labels)
    labels_count_sorted = sorted(labels_count.items(), key=lambda d:d[1], reverse = True)
    if len(labels_count) == 1:        # 即该（子）数据集中只有一类
        return labels[0]
    if len(dataSet[0]) == 1:          # 即该（子）数据集中特征为空
        return labels_count_sorted.values()[0]
    best_chara_index = chooseBestChara(dataSet, charas)
    best_chara = charas[best_chara_index]
    bestCharas.append(best_chara)
    tree = {best_chara:{}}
    del(charas[best_chara_index])
    chara_values = [data[best_chara_index] for data in dataSet]
    unique_vals = set(chara_values)
    for val in unique_vals:
        charas_sub = charas[:]
        tree[best_chara][val] = generateTree(splitDataSet(dataSet, best_chara_index, val), charas_sub, bestCharas)
    return tree

if __name__ == "__main__":

    dataSet, charas = createDataSet()
    best_charas = []
    print(generateTree(dataSet, charas, best_charas))