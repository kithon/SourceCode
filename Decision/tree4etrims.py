# -*- coding: utf-8 -*-
import os
import sys
import datetime
import numpy as np
import collections
from PIL import Image
from extreme import StackedELMAutoEncoder
        
##########################################################
##  Decision Tree (for etrims)
##########################################################

class DecisionTree(object):
    def __init__(self, radius=None, num_function=10,
                 condition='gini', seed=123):
        #print "Initialize decision tree"
        # define_range = [0. 255.]
        if radius is None:
            Exception('Error: radius is None.')
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
                
    def fit(self, picture, d_limit=None):
        #print "Fit"
        self.picture = picture
        input = []
        for i,p in enumerate(picture):
            w,h = p.getSize()
            input += [[i,j,k] for j in range(w) for k in range(h)]
        print input
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
            #print self.depth, ":[", len(l_data), len(r_data), "]"
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
        #print self.depth, len(l_data), len(r_data)
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

"""

##########################################################
##  ExtremeDecision Tree (for etrims)
##########################################################

class ExtremeDecisionTree(DecisionTree):
    def __init__(self, elm_hidden=None, elm_coef=None, define_range=[0., 1.],
                 radius=None, num_function=10, condition='gini', seed=123, visualize=False):
        DecisionTree.__init__(self, define_range, radius, num_function, condition, seed)
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
                
                #selmae = StackedELMAutoEncoder(n_hidden=n_hidden, coef=coef)
                #selmae.fit(data)
                #input = selmae.extraction(input)
                #print input
                
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
            
            #selected_dim = self.np_rng.randint(self.elm_hidden[-1])
            #theta = self.np_rng.rand()
            
            #print "gen:", "selected_dim", selected_dim,"theta", theta
            yield elm_threshold(selected_dim, theta, self.elm_hidden, self.elm_coef)
            
"""    
    
##########################################################
##  Experiment for etrims
##########################################################

class Pic(object):
    def __init__(self, data, signal):
        self.data = data
        self.signal = signal
        self.w, self.h = self.data.size

    def getSize(self):
        return self.w, self.h

    def getData(self, x, y):
        if x < 0 or x >= self.w:
            # out of x_range
            return 0
        if y < 0 or y >= self.h:
            # out of y_range
            return 0
        # in range
        return self.data.getpixel((x, y))

    def getSignal(self, x, y):
        # in range
        return self.signal.getpixel((x, y))

    
def print_time(message):
    d = datetime.datetime.today()
    print '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, message)    

    
def load_etrims(is08=True, size=6, shuffle=True, visualize=True):
    # path
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/' if is08 else '04_etrims-ds/'
        
    # train index
    train_index = []
    DATA_SIZE = 60
    TRAIN_SIZE = 40
    if shuffle:
        # shuffle train index
        if DATA_SIZE < TRAIN_SIZE:
            raise Exception('DATA_SIZE < TRAIN_SIZE')
        while len(train_index) < TRAIN_SIZE:
            tmp = np.random.randint(DATA_SIZE)
            if not tmp in train_index:
                train_index.append(tmp)
    else:
        # unshuffle train index
        train_index = range(TRAIN_SIZE)
    
    # path
    an_path = root_path + an_name + et_name
    im_path = root_path + im_name + et_name
    dir_list = os.listdir(an_path)

    # set
    train_set = []
    test_set = []
    test_or_train = [test_set, train_set]
        
    print "image datas and annotations..."
    for i, dis in enumerate(dir_list):
        # get filename, annotation_path and image_path
        file_name = dis.split(".")[0]
        annot_path = an_path + file_name + ".png"
        annotation = Image.open(annot_path)
        image_path = im_path + file_name + ".jpg"
        image = Image.open(image_path)
        
        # get index and set picture
        index = i in train_index
        picture = Pic(image, annotation)
        test_or_train[index].append(picture)

        # print filename
        print file_name, "done."

    print "train:", len(train_set), "test:", len(test_set)
    return train_set, test_set


def etrims_tree(n_hidden = [1000], coef = [1000.], size=6):
    print_time('load_etrims')
    train_set, test_set = load_etrims(size=size)

    num_function = 100
    print_time('tree2etrims test size is %d' % size)

    print_time('train_DecisionTree number of function is %d' % num_function)
    dt = DecisionTree(radius=size, num_function=num_function)
    dt.fit(train_set)

    print_time('test_DecisionTree')
    score = dt.score(test_set)
    print_time('score is %f' % score)

    print_time('DecisionTree info')
    dt.info()

    return ################################# debug ########################################


    elm_hidden = [(2*size+1)*(2*size+1)*2]

    print_time('train_ExtremeDecisionTree elm_hidden is %d, num function is %d' % (elm_hidden[0], num_function))
    edt = ExtremeDecisionTree(radius=size, elm_hidden=elm_hidden, elm_coef=None, num_function=num_function)
    edt.fit(train_set)

    print_time('test_ExtremeDecisionTree')
    score = edt.score(test_set)
    print_time('score is %f' % score)

    print_time('ExtremeDecisionTree info')
    edt.info()

    print_time('tree2etrims test is finished !')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        size = 6
        print "############ warning: size is forcely", size, '#############'
    else:
        size = int(sys.argv[1])
    etrims_tree(size=size)
    
