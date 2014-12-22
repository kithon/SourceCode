# -*- coding: utf-8 -*-
import numpy as np
import collections
from extreme import StackedELMAutoEncoder
        
##########################################################
##  Decision Tree
##########################################################

class DecisionTree(object):
    def __init__(self, define_range=[0., 1.], num_function=10,
                 condition='gini', seed=123):
        #print "Initialize decision tree"
        self.define_range = define_range
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)

    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        def threshold(selected_dim, theta):
            def function(input):
                #print "thre:", "selected_dim", selected_dim,"theta", theta
                return input[selected_dim] - theta
            return function

        numpy_data = np.array(data)
        for i in xrange(self.num_function):
            
            selected_dim = self.np_rng.randint(self.dim)
            selected_row = numpy_data.T[selected_dim]
            min_row = selected_row.min()
            max_row = selected_row.max()
            theta = self.np_rng.rand() * (max_row - min_row) + min_row
            #print "gen:", "selected_dim", selected_dim,"theta", theta
            """
            selected_dim = self.np_rng.randint(self.dim)
            theta = self.np_rng.rand()
            """
            #print "gen:", "selected_dim", selected_dim,"theta", theta
            yield threshold(selected_dim, theta)

    def normalize_input(self, input):
        #print "Normalize input"
        input = np.array(input)
        min_input = min(input.min(), self.define_range[0])
        max_input = max(input.max(), self.define_range[1])
        if max_input == min_input:
            return input.tolist()
        data = 1. * (input - min_input)
        data = data / (max_input - min_input)
        return data.tolist()
        
    def normalize_signal(self, signal):
        #print "Normalize signal"
        label_index, label_type = [], []
        for s in signal:
            if not s in label_type:
                label_type.append(s)
            label_index.append(label_type.index(s))
        self.label_type = label_type
        return label_index
        
    def fit(self, input, signal, d_limit=None):
        #print "Fit"
        data = self.normalize_input(input)
        label = self.normalize_signal(signal)
        self.dim = len(data[0])
        self.tree = Tree(data, label, self.generate_threshold, d_limit)

    def predict(self, input):
        #print "Predict"
        data = self.normalize_input(input)
        predict_signal = []
        for d in data:
            predict_signal.append(self.label_type[self.tree.predict(d)])
        return predict_signal

    def score(self, input, signal):
        #print "score"
        count = 0
        length = len(signal)
        data = self.normalize_input(input)
        for i in xrange(length):
            predict_signal = self.label_type[self.tree.predict(data[i])]
            if predict_signal == signal[i]:
                count += 1
        return count * 1.0 / length

    def info(self):
        print "Information"
        print "root node",
        depth_array = self.tree.info()
        print "depth", depth_array
        print "max depth", max(depth_array)
    
##########################################################
##  ExtremeDecision Tree
##########################################################

class ExtremeDecisionTree(DecisionTree):
    def __init__(self, elm_hidden=None, elm_coef=None, define_range=[0., 1.],
                 num_function=10, condition='gini', seed=123, visualize=False):
        DecisionTree.__init__(self, define_range, num_function, condition, seed)
        self.elm_hidden = elm_hidden
        self.elm_coef = elm_coef
        self.visualize = visualize
        
    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        selmae = StackedELMAutoEncoder(n_hidden=self.elm_hidden, coef=self.elm_coef, visualize=self.visualize)
        selmae.fit(data)
        def elm_threshold(selected_dim, theta, n_hidden, coef):
            def function(input):
                #print "thre:", "selected_dim", selected_dim,"theta", theta
                """
                selmae = StackedELMAutoEncoder(n_hidden=n_hidden, coef=coef)
                selmae.fit(data)
                input = selmae.extraction(input)
                print input
                """
                input = selmae.extraction(input)
                return input[selected_dim] - theta
            return function

        numpy_data = np.array(selmae.extraction(data))
        for i in xrange(self.num_function):
            
            selected_dim = self.np_rng.randint(self.elm_hidden[-1])
            selected_row = numpy_data.T[selected_dim]
            min_row = selected_row.min()
            max_row = selected_row.max()
            theta = self.np_rng.rand() * (max_row - min_row) + min_row
            #print "gen:", "selected_dim", selected_dim,"theta", theta
            """
            selected_dim = self.np_rng.randint(self.elm_hidden[-1])
            theta = self.np_rng.rand()
            """
            #print "gen:", "selected_dim", selected_dim,"theta", theta
            yield elm_threshold(selected_dim, theta, self.elm_hidden, self.elm_coef)
            

    """ 
    def normalize_input(self, input):
    
    def normalize_signal(self, signal):
    
    def fit(self, input, signal, d_limit=None):
    
    def predict(self, input):
    
    def score(self, input, signal):
    
    """
    
##########################################################
##  Tree
##########################################################

class Tree(object):
    def __init__(self, data, label, gen_threshold=None, d_limit=None, depth=0, condition='gini'):
        if gen_threshold is None:
            Exception("Error: Threshold generator is not defined.")

        self.depth = depth
        self.condition = condition
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
            l_data, l_label, r_data, r_label = [], [], [], []
            while len(l_data) == 0 or len(r_data) == 0:
                thresholds = [t for t in gen_threshold(data)]
                self.function = self.opt_threshold(data, label, thresholds)

                # divide
                l_data, l_label, r_data, r_label = self.divide(data, label, self.function)
                #print "len", len(l_data), len(r_data)
            self.l_tree = Tree(l_data, l_label, gen_threshold, d_limit, depth+1)
            self.r_tree = Tree(r_data, r_label, gen_threshold, d_limit, depth+1)

    def divide(self, data, label, function):
        lr_data = [[], []]
        lr_label = [[], []]
        for i, d in enumerate(data):
            index = (function(d) > 0)
            lr_data[index].append(d)
            lr_label[index].append(label[i])
            #print lr_label, index, label, i
            
        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def opt_threshold(self, data, label, thresholds):
        cost = self.gini if self.condition == 'gini' else self.entropy
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
                #print "sub", sub * p * (1. - p) 
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

    def info(self, depth=[]):
        #print "depth:", self.depth,
        if self.terminal:
            #print "terminal."
            return [self.depth]
        #print "left_node", 
        depth = depth + self.l_tree.info()
        #print "right_node",
        depth = depth + self.r_tree.info()
        return depth
        
if __name__ == '__main__':
    """
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]
    """
    
    train = [[1, 1], [1, -1], [-1, 1], [-1, -1], [0.1, 0.1], [0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1]]
    label = [1, -1, -1, 1, 1, -1, -1, 1]
    test = [[0.5, 1], [-0.5, -0.5], [0.3, -0.2], [-0.1, 0.9]]
    
    model = ExtremeDecisionTree(elm_hidden=[3,4,5,6], elm_coef=[1000., 1000., 1000., 1000.])
    #model = DecisionTree()

    model.fit(train, label)

    print "predict train"
    print model.predict(train)

    print "predict test"
    print model.predict(test)

    model.info()
