# -*- coding: utf-8 -*-
import os
import sys
import random
import datetime
import linecache
import numpy as np
import collections
import multiprocessing
from ast import literal_eval
from PIL import Image
from extreme import StackedELMAutoEncoder

DATA = datetime.datetime.today()
SUFFIX = 'parameter.log'
#SUFFIX =  '%s_%s_%s_%s.log' % (DATA.month, DATA.day, DATA.hour, DATA.minute)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

"""
def fit_process(dic, index, node):
    node.fit()
    parameter = node.save()
    dic.update({index: parameter})
"""

def fit_process(dic, index_list, node_list):
    for i,node in enumerate(node_list):
        node.fit()
        parameter = node.save()
        dic.update({index_list[i]: parameter})

    
##########################################################
##  Decision Tree (for etrims)
##########################################################

class DecisionTree(object):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'file_name',
                 'picture', 'node_length', 'parameter_list']
    def __init__(self, radius=None, num_function=10,
                 condition='gini', seed=123):
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)
        self.file_name = 'dt_' + SUFFIX

    def getNode(self, data=None, depth=None):
        return Node(data, self.picture, depth, self.generate_threshold, self.d_limit, self.condition)
            
    def fit(self, picture, d_limit=None):
        # initialize input
        input = []
        self.picture = picture
        self.d_limit = d_limit
        for i,p in enumerate(picture):
            w,h = p.getSize()
            input += [[i,j,k] for j in range(w) for k in range(h)]

        # initialize execution_list and wait_list
        exec_list = [self.getNode(input, 0)]
        wait_list = []
        node_length = len(exec_list)

        # initialize current_depth
        current_depth = 0
        core = multiprocessing.cpu_count()

        # open file to write
        f = open(self.file_name, 'w')
        while len(exec_list):
            # initialize jobs and dic
            jobs = []
            dic = multiprocessing.Manager().dict()
            num_process = min(core, len(exec_list))

            # print depth
            print_time("depth:%d" % current_depth)

            # distribute exec_list
            for i in xrange(num_process):
                index_list = range(i, len(exec_list), num_process)
                node_list = [exec_list[k] for k in index_list]
                # append process
                jobs.append(multiprocessing.Process(target=fit_process, args=(dic, index_list, node_list)))

            # multiprocessing
            for j in jobs:
                j.start()

            for j in jobs:
                j.join()

            # set parameter
            for i,node in enumerate(exec_list):
                parameter = dic.get(i)
                node.load(parameter)

            # make child node
            for node in exec_list:
                if not node.isTerminal():
                    # not terminal
                    node.setChildIndex(node_length)
                    node_length += 2
                    l_data, l_label, r_data, r_label  = node.divide()
                    # append l_node and r_node to wait_list
                    depth = node.getDepth() + 1
                    wait_list.append(self.getNode(l_data, depth))
                    wait_list.append(self.getNode(r_data, depth))

                # write node's data in self.dile_name
                f.write(str(node.save()) + '\n')
            
            # update execution_list, wait_list, node_length and current_depth
            exec_list = wait_list
            wait_list = []
            current_depth += 1

        # set node_length
        self.node_length = node_length
        # close file
        f.close()

    def generate_threshold(self, data):
        for i in xrange(self.num_function):
            # default radius (-6 <= x,y <= 6), channel (0 <= c < 3)
            selected_dx = self.np_rng.randint(-1*self.radius, self.radius+1)
            selected_dy = self.np_rng.randint(-1*self.radius, self.radius+1)
            selected_c = self.np_rng.randint(3)
            
            min_row = 0
            max_row = 255
            
            theta = self.np_rng.rand() * (max_row - min_row) + min_row
            
            selected_dim = [selected_dx, selected_dy, selected_c]
            yield selected_dim, theta

        
    def predict(self, data):
        #print "Predict"
        index = 0
        while True:
            node = self.getNode()
            node.setPicture(self.picture)
            parameter = literal_eval(linecache.getline(self.file_name, index+1).split('\n')[0])
            node.load(parameter)
            
            if node.isTerminal():
                #print "predict done"
                return node.predict(data)
            index = node.predict(data)

    def score(self, picture):
        # read parameter_list
        """
        f = open(self.file_name, 'r')
        str_list = f.read().split('\n')[0:self.node_length]
        self.parameter_list = map(literal_eval, str_list)
        f.close()
        """

        #print "score"
        self.picture = picture
        input = []
        for i,p in enumerate(picture):
            w,h = p.getSize()
            input += [[i,j,k] for j in range(w) for k in range(h)]
        count = 0
        length = len(input)
        for temp in input:
            i,x,y = temp
            predict_signal = self.predict(temp)
            if predict_signal == self.picture[i].getSignal(x,y):
                count += 1
        return count * 1.0 / length

    def info(self):
        print "Information"
        # need to revise
        print "node size", self.node_length

        
##########################################################
##  Node (for etrims)
##########################################################

class Node(object):
    __slots__ = ['data', 'picture', 'depth', 'gen_threshold',
                 'd_limit', 'condition', 'l_index', 'r_index',
                 'terminal', 'label', 'selected_dim', 'theta']
    def __init__(self, data=None, picture=None, depth=None, gen_threshold=None, d_limit=None, condition=None):
        if not data is None:
            self.data = data
            self.picture = picture
            self.depth = depth
            self.gen_threshold = gen_threshold
            self.d_limit = d_limit
            self.condition = condition
            self.l_index = 0
            self.r_index = 0

    def fit(self):
        #print "label", label
        label = []
        for temp in self.data:
            #print temp
            i, x, y = temp
            label.append(self.picture[i].getSignal(x,y))

        if len(set(label)) == 1:
            # terminate
            #print "terminate"
            self.terminal = True
            self.label = label[0]

        elif not self.d_limit is None and self.d_limit <= self.depth:
            # forcely terminate
            #print "break"
            self.terminal = True
            self.label = collections.Counter(label).most_common()[0][0]

        else:
            # continue
            self.terminal = False
            l_data, r_data = [], []
            while len(l_data) == 0 or len(r_data) == 0:
                #print "divide"
                thresholds = [t for t in self.gen_threshold(self.data)]
                #print "opt"
                self.opt_threshold(self.data, thresholds)

                #print "function"

                # divide
                l_data, l_label, r_data, r_label = self.divide()
                #print "len", len(l_data), len(r_data)
            #print self.depth, ":[", len(l_data), len(r_data), "]"

    def opt_threshold(self, data, thresholds):
        cost = self.gini if self.condition == 'gini' else self.entropy
        index = None
        minimum = None
        for i,threshold in enumerate(thresholds):
            # threshold consits of (selected_dim, theta)
            l_data, l_label, r_data, r_label = self.divide(data, threshold)
            temp = cost(l_label, r_label)
            if minimum is None or temp < minimum:
                index = i
                minimum = temp
        self.selected_dim, self.theta = thresholds[index]

    def divide(self, data=None, threshold=None):
        # set data
        if data is None:
            data = self.data

        # divide data and label
        lr_data = [[], []]
        lr_label = [[], []]
        for i, element in enumerate(data):
            #print element
            index = (self.function(element, threshold) > 0)
            lr_data[index].append(element)
            lr_label[index].append(self.picture[element[0]].getSignal(element[1], element[2]))
            #print lr_label, index, label, i

        l_data, r_data = lr_data
        l_label, r_label = lr_label
        #print self.depth, len(l_data), len(r_data)
        return l_data, l_label, r_data, r_label
        
    def function(self, element, threshold=None):
        # set threshold
        if threshold is None:
            selected_dim = self.selected_dim
            theta = self.theta
        else:
            selected_dim, theta = threshold

        i,  x,  y = element
        dx, dy, c = selected_dim
        #print element
        return self.picture[i].getData(x+dx, y+dy)[c] - theta

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

    
    def predict(self, element):
        # check terminal
        if self.terminal:
            return self.label

        # check threshold
        if self.function(element) > 0:
            return self.r_index
        else:
            return self.l_index

    def isTerminal(self):
        return self.terminal
    
    def getDepth(self):
        return self.depth
    
    def setPicture(self, picture):
        self.picture = picture

    def setChildIndex(self, index):
        # set child index
        self.l_index = index
        self.r_index = index+1
        
    def save(self):
        # return parameter
        detail = self.label if self.isTerminal() else [self.l_index, self.r_index, self.selected_dim, self.theta]
        parameter = [self.depth, self.d_limit, self.terminal, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, detail = parameter
        if self.isTerminal():
            self.label = detail
        else:
            self.l_index, self.r_index, self.selected_dim, self.theta = detail
        

##########################################################
##  ExtremeDecision Tree (for etrims)
##########################################################

class ExtremeDecisionTree(DecisionTree):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'file_name',
                 'picture', 'node_length', 'parameter_list',
                 'elm_hidden', 'elm_coef', 'visualize']
    def __init__(self, elm_hidden=None, elm_coef=None,
                 radius=None, num_function=10, condition='gini', seed=123, visualize=False):
        DecisionTree.__init__(self, radius, num_function, condition, seed)
        self.elm_hidden = elm_hidden
        self.elm_coef = elm_coef
        self.visualize = visualize
        self.file_name = 'edt_' + SUFFIX

    def getNode(self, data=None, depth=None):
        return ExtremeNode(data, self.picture, depth, self.generate_threshold, self.d_limit, self.radius, self.condition)
    
    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        selmae = StackedELMAutoEncoder(n_hidden=self.elm_hidden, coef=self.elm_coef, visualize=self.visualize)
        sample = []
        num = min(len(data), (2*self.radius+1)*(2*self.radius+1))
        sample_index = random.sample(data, num)
        for temp in sample_index:
            i,x,y = temp
            sample.append(self.picture[i].cropData(x, y, self.radius))
        betas, biases = selmae.fit(sample)

        numpy_data = np.array(selmae.extraction(sample))
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
            yield selected_dim, theta, betas, biases


##########################################################
##  Extreme Node (for etrims)
##########################################################

class ExtremeNode(Node):
    __slots__ = ['data', 'picture', 'depth', 'gen_threshold',
                 'd_limit', 'radius', 'condition', 'l_index', 'r_index',
                 'terminal', 'label', 'selected_dim', 'theta',
                 'betas', 'biases']
    def __init__(self, data, picture, depth, gen_threshold, d_limit, radius, condition):
        Node.__init__(self, data, picture, depth, gen_threshold, d_limit, condition)
        self.radius = radius

    def opt_threshold(self, data, thresholds):
        cost = self.gini if self.condition == 'gini' else self.entropy
        index = None
        minimum = None
        for i,threshold in enumerate(thresholds):
            # threshold consits of (selected_dim, theta, betas, biases)
            #print "size of threshold:", len(threshold)
            l_data, l_label, r_data, r_label = self.divide(data, threshold)
            temp = cost(l_label, r_label)
            if minimum is None or temp < minimum:
                index = i
                minimum = temp
        self.selected_dim, self.theta, self.betas, self.biases = thresholds[index]
        
    def function(self, element, threshold=None):
        # set threshold
        if threshold is None:
            selected_dim = self.selected_dim
            theta = self.theta
            betas = self.betas
            biases = self.biases
        else:
            selected_dim, theta, betas, biases = threshold
            
        i,  x,  y = element
        crop = self.picture[i].cropData(x, y, self.radius)
        for i, beta in enumerate(betas):
            bias = biases[i]
            crop = sigmoid(np.dot(crop, beta.T) + bias)
        return crop[selected_dim] - theta
    

    def save(self):
        # return parameter
        detail = self.label if self.isTerminal() else [self.l_index, self.r_index,
                                                       self.selected_dim, self.theta,
                                                       map(lambda n:n.tolist(), self.betas),
                                                       map(lambda n:n.tolist(), self.biases)]
        parameter = [self.depth, self.d_limit, self.terminal, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, detail = parameter
        if self.isTerminal():
            self.label = detail
        else:
            l, r, s, t, be, bi = detail
            self.l_index = l
            self.r_index = r
            self.selected_dim = s
            self.theta = t
            self.betas = map(np.array, be)
            self.biases = map(np.array, bi)
            
    
    
##########################################################
##  Experiment for etrims
##########################################################

class Pic(object):
    __slots__ = ['data', 'signal', 'w', 'h']
    def __init__(self, data, signal):
        self.w, self.h = data.size
        self.setData(data)
        self.setSignal(signal)

    def setData(self, data):
        data_list = []
        for x in xrange(self.w):
            row = []
            for y in xrange(self.h):
                row.append(list(data.getpixel((x,y))))
            data_list.append(row)
        self.data = data_list
        
    def setSignal(self, signal):
        signal_list = []
        for x in xrange(self.w):
            row = []
            for y in xrange(self.h):
                row.append(signal.getpixel((x,y)))
            signal_list.append(row)
        self.signal = signal_list
        
    def getSize(self):
        return self.w, self.h

    def getData(self, x, y):
        if x < 0 or x >= self.w:
            # out of x_range
            return [0,0,0]
        if y < 0 or y >= self.h:
            # out of y_range
            return [0,0,0]
        # in range
        return self.data[x][y]

    def getSignal(self, x, y):
        # in range
        return self.signal[x][y]

    def cropData(self, x, y, radius):
        crop = []
        for dx in range(x-radius, x+radius+1):
            for dy in range(y-radius, y+radius+1):
                crop += self.getData(dx, dy)
        crop = (1. * np.array(crop) / 255).tolist()
        return crop
    
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
    DATA_SIZE = 6 # 60 ########debug
    TRAIN_SIZE = 4 # 40 ########debug
    if shuffle:
        # shuffle train index
        train_index = random.sample(range(DATA_SIZE), TRAIN_SIZE)
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
    for i in xrange(DATA_SIZE):
        dis = dir_list[i]
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


def etrims_tree(n_hidden = [1000], coef = [1000.], size=6, d_limit=None):
    print_time('load_etrims')
    train_set, test_set = load_etrims(size=size)

    num_function = 10 #100 ##########debug#####################
    print_time('tree2etrims test size is %d' % (size))

    print_time('train_DecisionTree number of function is %d' % num_function)
    dt = DecisionTree(radius=size, num_function=num_function)
    dt.fit(train_set, d_limit=d_limit)

    print_time('test_DecisionTree')
    score = dt.score(test_set)
    print_time('score is %f' % score)

    print_time('DecisionTree info')
    dt.info()
    
    #return ################################# debug ########################################
    
    
    elm_hidden = [3 * (2*size+1)*(2*size+1) * 2] # (3 channel) * (width) * (height) * 2

    print_time('train_ExtremeDecisionTree elm_hidden is %d, num function is %d' % (elm_hidden[0], num_function))
    edt = ExtremeDecisionTree(radius=size, elm_hidden=elm_hidden, elm_coef=None, num_function=num_function)
    edt.fit(train_set, d_limit=d_limit)

    print_time('test_ExtremeDecisionTree')
    score = edt.score(test_set)
    print_time('score is %f' % score)

    print_time('ExtremeDecisionTree info')
    edt.info()

    print_time('tree2etrims test is finished !')


if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) == 1:
        size = 6
        print "############ warning: size is forcely", size, '#############'
        etrims_tree(size=size)
    elif len(sys.argv) == 2:
        size = int(sys.argv[1])
        etrims_tree(size=size)    
    else:
        size = int(sys.argv[1])
        d_limit = int(sys.argv[2])
        etrims_tree(size=size, d_limit=d_limit)
    
