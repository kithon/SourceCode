# -*- coding: utf-8 -*-
import numpy as np
import collections

def DecisionTree(object):
    def __init__(self):
        print "init"
        self.dim = 100

    def generate_threshold(size=10):
        def threshold(selected_dim, theta):
            def function(input):
                return input[selected_dim] - theta
            return function
        
        for i in xrange(size):
            selected_dim = np.random ###### 0 ~ (self.dim-1)
            theta = np.random ####### 0. ~ 1.
            yield threshold(selected_dim, theta)

    def normalize_input(input):
        print "normalization"
    def normalize_signal(signal):
        print "normalization"
        
    def fit(self, input, signal):
        print "fit"
        data = normalize_input(input)
        label = normalize_signal(signal)

        self.tree = Tree(data, label, self.generate_threshold, d_limit=None)
        



    def predict(self, input):
        print "predict"
        

def Tree(object):
    def __init__(self, data, label, gen_threshold=None, d_limit=None, depth=0):
        if gen_threshold is None:
            Exception("Error: Threshold generator is not defined.")

        if len(set(label)) == 1:
            # terminate
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
            self.function = self.opt_threshold(thresholds)

            # divide
            l_data, l_label, r_data, r_label = self.divide(data, label, self.function)
            self.l_tree(l_data, l_label, gen_threshold, d_limit, depth+1)
            self.r_tree(r_data, r_label, gen_threshold, d_limit, depth+1)

    def divide(self, data, label, function):
        lr_data = [[], []]
        lr_label = [[], []]
        for i, d in enumerate(data):
            index = function(input) > 0
            lr_data[index].append(d)
            lr_label[index].append(label[i])
            
        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def opt_threshold(self, threshold):
        print "optimization"
            
    def gini(self, data):
        # get gini
        print "gini"

    def entropy(self, data):
        # get entropy
        print "entropy"

    def predict(self, input):
        # check terminal
        if self.terminal:
            return self.label

        # check threshold
        if self.function(input) > 0:
            return self.r_tree.predict(input)
        else:
            return self.l_tree.predict(input)
        
        
if __name__ == '__main__':
    print "decision"
