#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

class SVM(object):

    def __init__(self, dataset, labels, C, tol = 0.001) -> None:
        super().__init__()
    
        self.dataset = dataset  
        self.labels = labels
        self.C = C              
        self.tol = tol                  # tolerance 松弛变量

        self.m, self.n = np.array(self.dataset).shape 

        self.alpha = np.zeros(self.m)   # init alphas
        self.b = 0                      # init b

        self.error = [self.get_error for i in range(self.m)]

    def predict_func(self, x):
        '''
        f(x) = wT * x + b
        w = Σ alpha_i * y_i * x_i    (KKT)
          = alpha * label * dataset
        dataset:[[x_1],
                 [x_2],
                 ...
                 [x_n]
                 ]
        alpha * labels = [a_1*y_1, a_2*y_2, ..., a_n*y_n]
        SVM classifier predict
        '''
        x = np.matrix(x).T
        dataset = np.matrix(self.dataset)
        w = np.matrix(self.alpha * self.labels) * dataset
        return  w * x + self.b

    def get_error(self, i):
        '''
        Error = predict_func(x_i) - y_i
        '''
        x, y = self.dataset[i], self.labels[i]
        return self.predict_func(x) - y

    def update_error(self):

        self.error = [self.get_error for i in range(self.m)]

    def get_w(self):

        return (np.matrix(self.alpha * self.labels) * np.matrix(self.dataset)).tolist()

    def update_alpha(self, a, i):

        self.alpha[i] = a
    
    def kkt(self, i):
        '''
        Determine whether it meets KKT conditions
        '''
        a = self.alpha[i]
        x = self.dataset[i]
        if a == 0:
            return
        elif a == self.C:
            return
        else:
            return

def load_data(filename):

    dataset = []
    labels = []

    with open(file=filename, mode='r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels

def clip(a, L, H):
    '''
    According to KKT  clip alpha
    '''
    if a < L:
        return L
    elif a > H:
        return H
    else:
        return a

def select_rand_j(i, m):

    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)

def opti_step(i, j, svm_util:SVM) ->SVM:
    
    svm_util.update_error()
    alpha, dataset, labels, b = svm_util.alpha, svm_util.dataset, svm_util.labels, svm_util.b
    a_i, a_j = alpha[i], alpha[j]
    x_i, x_j = dataset[i], dataset[j]
    y_i, y_j = labels[i], labels[j]

    K_ii = np.dot(x_i, x_i)
    K_jj = np.dot(x_j, x_j)
    K_ij = np.dot(x_i, x_j)

    f_xi = svm_util.predict_func(x_i)
    f_xj = svm_util.predict_func(x_j)

    v1 = f_xi - a_i * y_i * K_ii - a_j * y_j * K_ij - b
    v2 = f_xj - a_i * y_i * K_ij - a_j * y_j * K_jj - b

    E_i = f_xi - y_i
    E_j = f_xj - y_j

    eta = K_ii + K_jj - 2 * K_ij

    a_i_old, a_j_old = a_i, a_j
    a_j_new = a_j_old + y_j * (E_i - E_j) / eta

    # clip

    if y_i == y_j:
        L = max(0, a_i_old + a_j_old - svm_util.C)
        H = min(svm_util.C, a_i_old + a_j_old)
    else:
        L = max(0, a_j_old - a_i_old)
        H = min(svm_util.C, svm_util.C + a_j_old - a_i_old)

    a_j_new = clip(a_j_new, L, H)
    a_i_new = a_i_old + (a_j_old - a_j_new) * y_i * y_j

    '''if abs(a_j_new - a_j_old) < 0.00001:
        #print('WARNING   alpha_j not moving enough')
        return 0'''

    svm_util.update_alpha(a_i_new, i)
    svm_util.update_alpha(a_j_new, j)
    svm_util.update_error()

    b_i = - E_i - y_i * K_ii * (a_i_new - a_i_old) \
        - y_j * K_ij * (a_j_new - a_j_old) + b
    b_j = - E_j - y_i * K_ij * (a_i_new - a_i_old) \
        - y_j * K_jj * (a_j_new - a_j_old) + b

    if 0 < a_i_new < svm_util.C:
        b = b_i
    elif 0 < a_j_new < svm_util.C:
        b = b_j
    else:
        b = (b_i + b_j) / 2
    
    svm_util.b = b

    return svm_util

def smo(dataset, labels, C, max_iter):

    svm_util = SVM(dataset, labels, C)
    cnt = 0
    pair_changed = 0

    while cnt < max_iter:
        pair_changed = 0
        for i in range(svm_util.m):
            j = select_rand_j(i, svm_util.m)
            svm_util = opti_step(i, j, svm_util)
            pair_changed += 1
            print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(cnt, i, pair_changed))
        
        if pair_changed != 0:
            cnt += 1
        else:
            cnt = 0
        print('iteration number: {}'.format(cnt))

            
    return svm_util


if __name__ == "__main__":

    dataset, labels = load_data('testSet.txt')
    svm_util = smo(dataset, labels, 0.6, 140)
    alpha, b = svm_util.alpha, svm_util.b

    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    # 绘制分割线
    w = svm_util.get_w()
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    b = np.double(b)
    a1, a2 = w[0]
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    print(x1, x2, y1, y2)
    ax.plot([x1, x2], [y1, y2])
    
    # 绘制支持向量
    for i, alpha in enumerate(alpha):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')

    plt.show()
