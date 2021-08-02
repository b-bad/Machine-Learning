import collections

class NAIVE_BAYES(object):

    def __init__(self, dataset, labels) -> None:
        super().__init__()

        self.dataset = dataset
        self.labels = labels
        self.labels_size = len(labels)
        self.x_dim = len(dataset[0]) - 1
        self.prior_proba = {}
        self.con_proba = {}
        self.labels_dict = collections.Counter(self.labels)
        self.prior_proba = self.get_prior_probability()
        self.con_proba = self.get_conditional_proability()

    def get_prior_probability(self):
        '''calculate prior probability'''
        
        for label in self.labels_dict:
            self.prior_proba[label] = float(self.labels_dict[label] / self.labels_size)
        
        return self.prior_proba

    def get_conditional_proability(self):
        '''calculate conditional probability'''

        dataset_c = self.dataset[:]

        for label_i in self.labels_dict:

            label_c_p = []
            data_label_i = []
            index = 0

            while index < len(dataset_c):

                if dataset_c[index][-1] == label_i:
                    data_label_i.append(dataset_c[index])
                    dataset_c.pop(index)
                else:
                    index += 1

            for i in range(self.x_dim):

                x_i = [data[i] for data in data_label_i]
                x_i_dict = collections.Counter(x_i)

                for x in x_i_dict:
                    x_i_dict[x] = float(x_i_dict[x] / self.labels_dict[label_i])
                label_c_p.append(x_i_dict)

            self.con_proba[label_i] = label_c_p

        return self.con_proba

    def predict(self, X):

        pre_proba = {}

        for label in self.labels_dict:
            proba = self.prior_proba[label]
            for i in range(self.x_dim):
                proba *= self.con_proba[label][i][X[i]]
            pre_proba[label] = proba
        
        pre_proba_s = sorted(pre_proba.items(), key=lambda d:d[1], reverse = True)

        return pre_proba_s[0]

def load_data(filename):

    dataset = []
    labels = []

    with open(file=filename, mode='r') as f:

        for line in f:
            data = [i for i in line.strip().split()]
            dataset.append(data)
            labels.append(data[-1])
            
    return dataset, labels


if __name__ == "__main__":

    dataset, labels = load_data("testSet.txt")
    NB = NAIVE_BAYES(dataset, labels)
    # print(NB.prior_proba)
    # print(NB.con_proba)
    print(NB.predict(['2', 'S']))
