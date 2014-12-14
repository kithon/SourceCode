# -*- coding: utf-8 -*-
import numpy as np
import collections

class DecisionTree(object):
    def __init__(self, seed=123):
        print "init"
        self.dim = 100
        self.np_rng = np.random.RandomState(seed)

    def generate_threshold(self, size=10):
        def threshold(selected_dim, theta):
            def function(input):
                return input[selected_dim] - theta
            return function
        
        for i in xrange(size):
            selected_dim = self.np_rng.randint(self.dim)
            theta = self.np_rng.rand()
            yield threshold(selected_dim, theta)

    def normalize_input(self, input):
        # input normalization
        input = np.array(input)
        min_input = input.min()
        max_input = input.max()
        if max_input == min_input:
            return input.tolist()
        data = 1. * (input - min_input)
        data = data / (max_input - min_input)
        return data.tolist()
        
    def normalize_signal(self, signal):
        # signal normalization
        label_index, label_type = [], []
        for s in signal:
            if not s in label_type:
                label_type.append(s)
            label_index.append(label_type.index(s))
        self.label_type = label_type
        return label_index
        
    def fit(self, input, signal):
        # fit
        data = self.normalize_input(input)
        label = self.normalize_signal(signal)
        self.dim = len(data[0])
        self.tree = Tree(data, label, self.generate_threshold, d_limit=None)

    def predict(self, input):
        # predict
        data = self.normalize_input(input)
        predict_signal = []
        for d in data:
            predict_signal.append(self.label_type[self.tree.predict(d)])
        return predict_signal

    def score(self, input, signal):
        # score
        count = 0
        length = len(signal)
        data = self.normalize_input(input)
        for i in xrange(length):
            predict_signal = self.label_type[self.tree.predict(data[i])]
            if predict_signal == signal[i]:
                count += 1
        return count * 1.0 / length
        

class Tree(object):
    def __init__(self, data, label, gen_threshold=None, d_limit=None, depth=0):
        if gen_threshold is None:
            Exception("Error: Threshold generator is not defined.")

        #print "label", label
        if len(set(label)) == 1:
            # terminate
            #print "terminate"
            self.terminal = True
            self.label = set(label).pop()

        elif d_limit >= depth:
            # forcely terminate
            self.terminal = True
            self.label = collections.Counter(label).most_common()[0][0]
        
        else:
            # continue
            self.terminal = False
            thresholds = [t for t in gen_threshold()]
            self.function = self.opt_threshold(data, label, thresholds)

            # divide
            l_data, l_label, r_data, r_label = self.divide(data, label, self.function)
            self.l_tree = Tree(l_data, l_label, gen_threshold, d_limit, depth+1)
            self.r_tree = Tree(r_data, r_label, gen_threshold, d_limit, depth+1)

    def divide(self, data, label, function):
        lr_data = [[], []]
        lr_label = [[], []]
        for i, d in enumerate(data):
            index = (function(d) > 0)
            lr_data[index].append(d)
            lr_label[index].append(label[i])
            
        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def opt_threshold(self, data, label, thresholds, condition='gini'):
        cost = self.gini if condition == 'gini' else self.entropy
        c_array = []
        for t in thresholds:
            l_data, l_label, r_data, r_label = self.divide(data, label, t)
            c_array.append(cost(l_label, r_label))
        index = c_array.index(min(c_array))
        return thresholds[index]
            
    def gini(self, l_label, r_label):
        # get gini (minimize)
        set_size = len(l_label) + len(r_label)
        g = 0
        for label in [l_label, r_label]:
            sub_size = len(label)
            counter = collections.Counter(label).most_common()
            for c in counter:
                p = 1. * c[1] / sub_size
                sub = (1. * sub_size / set_size)
                g += sub * p * (1. - p)
        return g
            
    def entropy(self, l_label, r_label):
        # get entropy (minimize)
        set_size = len(l_label) + len(r_label)
        e = 0
        for label in [l_label, r_label]:
            sub_size = len(label)
            counter = collections.Counter(label).most_common()
            for c in counter:
                p = 1. * c[1] / sub_size
                sub = (1. * sub_size / set_size)
                e += sub * p * np.log2(p)
        return -1. * e

    def predict(self, data):
        # check terminal
        if self.terminal:
            return self.label

        # check threshold
        if self.function(data) > 0:
            return self.r_tree.predict(data)
        else:
            return self.l_tree.predict(data)
        
        
if __name__ == '__main__':
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]

    model = DecisionTree()

    model.fit(train, label)

    print model.predict(train)
    print model.predict(test)
