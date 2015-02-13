# -*- coding: utf-8 -*-
import os
import random
import datetime
import linecache
import numpy as np
import collections
import multiprocessing
import argparse
from PIL import Image
from ast import literal_eval
from extreme import StackedELMAutoEncoder, BinaryELMClassifier

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

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
                 'np_rng', 'd_limit', 'dir_name', 'file_name',
                 'picture', 'node_length', 'parameter_list']
    def __init__(self, radius=None, num_function=10, condition='gini', seed=123, file_name=None):
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)
        self.dir_name = 'decision/'
        self.file_name = file_name

    def mkdir(self):
        path = self.dir_name
        if not os.path.exists(path):
            cmd = 'mkdir %s' % path
            os.system(cmd)
        self.clearFile()
        print_parameter(path, self.file_name)
    
    def getFileName(self, index):
        return '%snode_%d.log'% (self.dir_name, index)
    
    def clearFile(self):
        path = self.dir_name
        cmd = 'rm -f %s*' % path
        os.system(cmd)

    def getNode(self, data=None, depth=None):
        return Node(data, self.picture, depth, self.generate_threshold, self.d_limit, self.condition)
            
    def fit(self, picture, d_limit=None, overlap=True):
        # ----- initialize -----
        # -*- input, picture, param, d_limit -*-
        self.mkdir()
        self.picture = picture
        self.d_limit = d_limit
        input = []
        for i,p in enumerate(picture):
            w,h = p.getSize()
            if overlap:
                input += [[i,j,k] for j in range(w) for k in range(h)]
            else:
                input += [[i,j,k] for j in range(self.radius, w, 2*self.radius+1) for k in range(self.radius, h, 2*self.radius+1)]

        # -*- execution_list, wait_list, node_length -*-
        exec_list = [self.getNode(input, 0)]
        wait_list = []
        node_length = len(exec_list)

        # -*- current_depth, core -*-
        current_depth = 0
        core = multiprocessing.cpu_count()

        # ----- fit processing -----
        #f = open(self.file_name, 'w')
        index = 0
        while len(exec_list):
            # -*- initialize jobs, dic -*-
            jobs = []
            dic = multiprocessing.Manager().dict()
            num_process = min(core, len(exec_list))

            # -*- print depth -*-
            print_time("depth:%d" % current_depth, self.file_name)

            # -*- distribute exec_list -*-
            for i in xrange(num_process):
                index_list = range(i, len(exec_list), num_process)
                node_list = [exec_list[k] for k in index_list]
                jobs.append(multiprocessing.Process(target=fit_process, args=(dic, index_list, node_list)))

            # -*- multiprocessing -*-
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()

            # -*- set parameter -*-
            for i,node in enumerate(exec_list):
                parameter = dic.get(i)
                node.load(parameter)

            # -*- make child node -*-
            for node in exec_list:
                if not node.isTerminal():
                    node.setChildIndex(node_length)
                    node_length += 2
                    l_data, l_label, r_data, r_label  = node.divide()
                    depth = node.getDepth() + 1
                    wait_list.append(self.getNode(l_data, depth))
                    wait_list.append(self.getNode(r_data, depth))

                # -*- write node's data in self.dile_name -*-
                f = open(self.getFileName(index), 'w')
                f.write(str(node.save()))
                f.close()
                index += 1
                #f.write(str(node.save()) + '\n')
            
            # -*- update execution_list/wait_list/node_length/current_depth -*-
            exec_list = wait_list
            wait_list = []
            current_depth += 1

        # -*- set node_length -*-
        self.node_length = node_length
        #f.close()

    def generate_threshold(self, data):
        for i in xrange(self.num_function):
            # -*- channel (0 <= c < 3) -*-
            selected_dx = self.np_rng.randint(-1*self.radius, self.radius+1)
            selected_dy = self.np_rng.randint(-1*self.radius, self.radius+1)
            selected_c = self.np_rng.randint(3)
            
            min_row = 0
            max_row = 255
            
            theta = self.np_rng.rand() * (max_row - min_row) + min_row
            
            selected_dim = [selected_dx, selected_dy, selected_c]
            yield selected_dim, theta

    def predict(self, data):
        index = 0
        while True:
            node = self.getNode()
            node.setPicture(self.picture)
            parameter = literal_eval(linecache.getline(self.getFileName(index), 1))
            node.load(parameter)
            
            if node.isTerminal():
                #print "predict done"
                return node.predict(data)
            index = node.predict(data)

    def score(self, picture):
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
        self.clearFile()
        return count * 1.0 / length
    """
    def score(self, picture, d_limit=None):
        input = []
        self.picture = picture
        for i,p in enumerate(picture):
            w,h = p.getSize()
            input += [[i,j,k] for j in range(w) for k in range(h)]

        wait_list = []
        exec_list = [self.getNode(input, 0)]
    
        index = 0
        fix_count = 0
        current_depth = 0
        length = len(input)
        while len(exec_list):
            count = 0
            for i,node in enumerate(exec_list):
                parameter = literal_eval(linecache.getline(self.getFileName(index), 1))
                node.load(parameter)
                index += 1
                if not node.isTerminal():
                    count += node.getScore()
                    depth = node.getDepth() + 1
                    l_data, l_label, r_data, r_label  = node.divide()
                    wait_list.append(self.getNode(l_data, depth))
                    wait_list.append(self.getNode(r_data, depth))
                else:
                    fix_count += node.getScore()
                
            count += fix_count
            score = count * 1.0 / length
            print_time("depth:%d score = %f" % (current_depth, score), self.file_name)
            
            exec_list = wait_list
            wait_list = []
            current_depth += 1
    
        return score
    """
    
    def info(self):
        if not self.node_length is None:
            print_time("Information: number of node = %d" % (self.node_length), self.file_name)
        else:
            print_time("Information: self.node_length is not defined", self.file_name)

        
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

        self.label = collections.Counter(label).most_common()[0][0]
        if len(set(label)) == 1:
            self.terminal = True
        elif not self.d_limit is None and self.d_limit <= self.depth:
            self.terminal = True
        else:
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

    def getScore(self):
        score = 0
        for i, element in enumerate(self.data):
            i,x,y = element
            if self.picture[i].getSignal(x,y) == self.label:
                score += 1
        return score
        
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
        detail = [] if self.isTerminal() else [self.l_index, self.r_index, self.selected_dim, self.theta]
        parameter = [self.depth, self.d_limit, self.terminal, self.label, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, self.label, detail = parameter
        if not self.isTerminal():
            self.l_index, self.r_index, self.selected_dim, self.theta = detail
        

##########################################################
##  ExtremeDecision Tree (for etrims)
##########################################################

class ExtremeDecisionTree(DecisionTree):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'dir_name', 'file_name',
                 'picture', 'node_length', 'parameter_list',
                 'elm_hidden', 'elm_coef', 'visualize']
    def __init__(self, elm_hidden=None, elm_coef=None,
                 radius=1, num_function=10, condition='gini', seed=123, visualize=False, file_name=None):
        DecisionTree.__init__(self, radius, num_function, condition, seed, file_name)
        if elm_hidden is None:
            elm_hidden = [3 * (2*radius+1) * (2*radius+1) * 2]
        self.elm_hidden = elm_hidden
        self.elm_coef = elm_coef
        self.visualize = visualize
        self.dir_name = 'extreme/'
        

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
        detail = [] if self.isTerminal() else [self.l_index, self.r_index,
                                               self.selected_dim, self.theta,
                                               map(lambda n:n.tolist(), self.betas),
                                               map(lambda n:n.tolist(), self.biases)]
        parameter = [self.depth, self.d_limit, self.terminal, self.label, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, self.label, detail = parameter
        if not self.isTerminal():
            l, r, s, t, be, bi = detail
            self.l_index = l
            self.r_index = r
            self.selected_dim = s
            self.theta = t
            self.betas = map(np.array, be)
            self.biases = map(np.array, bi)
            

##########################################################
##  ExtremeBinaryDecision Tree (for etrims)
##########################################################

class BinaryExtremeDecisionTree(DecisionTree):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'dir_name', 'file_name',
                 'picture', 'node_length', 'parameter_list',
                 'elm_hidden', 'elm_coef', 'visualize']
    def __init__(self, elm_hidden=None, elm_coef=None,
                 radius=None, num_function=10, condition='gini', seed=123, visualize=False, file_name=None):
        DecisionTree.__init__(self, radius, num_function, condition, seed, file_name)
        if elm_hidden is None:
            elm_hidden = 3 * (2*radius+1) * (2*radius+1) * 2
        self.elm_hidden = elm_hidden
        self.elm_coef = elm_coef
        self.visualize = visualize
        self.dir_name = 'binary/'

    def getNode(self, data=None, depth=None):
        return BinaryExtremeNode(data, self.picture, depth, self.generate_threshold, self.d_limit, self.radius, self.condition)
    
    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        for i in xrange(self.num_function):            
            elm = BinaryELMClassifier(n_hidden=self.elm_hidden, coef=self.elm_coef)
            sample_input, sample_label  = [], []
            num = min(len(data), (2*self.radius+1)*(2*self.radius+1))

            label = set()
            while len(label) < 2:
                label.clear()
                sample_index = random.sample(data, num)
                for temp in sample_index:
                    i,x,y = temp
                    label.add(self.picture[i].getSignal(x, y))

            length = len(label) / 2
            one_index = random.sample(label, length)
            for temp in sample_index:
                i,x,y = temp
                sample_input.append(self.picture[i].cropData(x, y, self.radius))
                if self.picture[i].getSignal(x, y) in one_index:
                    sample_label.append(1)
                else:
                    sample_label.append(0)
                
            weight, bias, beta = elm.fit(sample_input, sample_label)
            yield weight, bias, beta


##########################################################
##  Binary Extreme Node (for etrims)
##########################################################

class BinaryExtremeNode(Node):
    __slots__ = ['data', 'picture', 'depth', 'gen_threshold',
                 'd_limit', 'radius', 'condition', 'l_index', 'r_index',
                 'terminal', 'label',
                 'weight', 'bias', 'beta']
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
        self.weight, self.bias, self.beta = thresholds[index]
        
    def function(self, element, threshold=None):
        # set threshold
        if threshold is None:
            weight = self.weight
            bias = self.bias
            beta = self.beta
        else:
            weight, bias, beta = threshold
            
        i, x, y = element
        crop = self.picture[i].cropData(x, y, self.radius)
        crop = sigmoid(np.dot(weight.T, crop) + bias)
        crop = np.dot(beta.T, crop)
        return crop - 0.5
    

    def save(self):
        # return parameter
        detail = [] if self.terminal else [self.l_index, self.r_index,
                                           self.weight.tolist(),
                                           self.bias.tolist(),
                                           self.beta.tolist()]
        parameter = [self.depth, self.d_limit, self.terminal, self.label, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, self.label, detail = parameter
        if not self.isTerminal():
            l, r, we, bi, be = detail
            self.l_index = l
            self.r_index = r
            self.weight = np.array(we)
            self.bias = np.array(bi)
            self.beta = np.array(be)            
    
    
##########################################################
##  Pic
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

##########################################################
##  print
##########################################################

"""
def print_parameter(param):
    cmd = 'echo %s >> %s' % (param, PAR_NAME)
    os.system(cmd)
    
def print_time(message):
    d = datetime.datetime.today()
    string = '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, message)
    cmd = 'echo %s >> %s' % (string, LOG_NAME)
    os.system(cmd)
"""

def print_parameter(param, FILE_NAME):
    cmd = 'echo %s >> %s' % (param, FILE_NAME)
    os.system(cmd)
    
def print_time(message, FILE_NAME):
    d = datetime.datetime.today()
    string = '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, message)
    cmd = 'echo %s >> %s' % (string, FILE_NAME)
    os.system(cmd)

        
##########################################################
##  load_etrims
##########################################################
    
def load_etrims(radius, size, is08, shuffle, name):
    # ----- path initialize -----
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/' if is08 else '04_etrims-ds/'
    an_path = root_path + an_name + et_name
    im_path = root_path + im_name + et_name
    dir_list = os.listdir(an_path)
        
    # ----- train index -----
    train_index = []
    DATA_SIZE = size # max=60 
    TRAIN_SIZE = 2 * size / 3 # max=40
    train_index = random.sample(range(DATA_SIZE), TRAIN_SIZE) if shuffle else range(TRAIN_SIZE)

    # ----- test set and train set -----
    train_set = []
    test_set = []
    test_or_train = [test_set, train_set]        
    for i in xrange(DATA_SIZE):
        # open annotation.png and image.jpg
        dis = dir_list[i]
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
        print_time("eTRIMS: %s" % file_name, name)

    # ----- finish -----
    print_parameter(train_index, name)
    print_time("eTRIMS: train=%d test=%d" % (len(train_set), len(test_set)), name)


def etrims_tree(radius, size, d_limit, unshuffle, four, num, parameter, t_args, file_name):
    # ----- initialize -----
    print_parameter([radius, size, d_limit, unshuffle, four, num, t_args], file_name)
    print_time('eTRIMS: radius=%d, depth_limit=%d, data_size=%d, num_func=%d' % (radius, d_limit, size, num), file_name)
    print_time('eTRIMS: load', file_name)
    train_set, test_set = load_etrims(radius=radius, size=size, is08=not four, shuffle=not unshuffle, name=file_name)
    isDT, isEDT, isBEDT = t_args
    
    # ----- Decision Tree -----
    if isDT:
        print_time('DecisionTree overlap: init', file_name)
        dt = DecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('DecisionTree overlap: train', file_name)
        dt.fit(train_set, d_limit=d_limit, overlap=True)
        
        print_time('DecisionTree overlap: test', file_name)
        score = dt.score(test_set)
        print_time('DecisionTree overlap: score = %f' % score, file_name)
        
        print_time('DecisionTree overlap: info', file_name)
        dt.info()

        
        print_time('DecisionTree: init', file_name)
        dt = DecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('DecisionTree: train', file_name)
        dt.fit(train_set, d_limit=d_limit, overlap=False)
        
        print_time('DecisionTree: test', file_name)
        score = dt.score(test_set)
        print_time('DecisionTree: score = %f' % score, file_name)
        
        print_time('DecisionTree: info', file_name)
        dt.info()
    

    # ----- Extreme Decision Tree -----
    if isEDT:
        print_time('ExtremeDecisionTree: init', file_name)
        edt = ExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('ExtremeDecisionTree: train', file_name)
        edt.fit(train_set, d_limit=d_limit, overlap=False)
        
        print_time('ExtremeDecisionTree: test', file_name)
        score = edt.score(test_set)
        print_time('ExtremeDecisionTree: score = %f' % score, file_name)
        
        print_time('ExtremeDecisionTree: info', file_name)
        edt.info()

    # ----- Binary Extreme Decision Tree -----
    if isBEDT:
        print_time('BinaryExtremeDecisionTree: init', file_name)
        bedt = BinaryExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('BinaryExtremeDecisionTree: train', file_name)
        bedt.fit(train_set, d_limit=d_limit, overlap=False)
        
        print_time('BinaryExtremeDecisionTree: test', file_name)
        score = bedt.score(test_set)
        print_time('BinaryExtremeDecisionTree: score = %f' % score, file_name)
        
        print_time('BinaryExtremeDecisionTree: info', file_name)
        bedt.info()


    # ----- finish -----
    print_time('eTRIMS: finish', file_name)


if __name__ == '__main__':
    # ----- parser description -----
    parser = argparse.ArgumentParser(description='Test eTRIMS-08 Segmentation Dataset (need etrims_tree.py)')
    parser.add_argument("name", type=str, default='result.log', help="set file name")
    parser.add_argument("radius", type=int, default=2, nargs='?', help="set image radius")
    parser.add_argument("size", type=int, default=60, nargs='?', help="set data size")
    parser.add_argument("limit", type=int, nargs='?', help="set depth limit")
    parser.add_argument("-r", "--removeparam", action='store_true',  help="remove parameter")
    parser.add_argument("-u", "--unshuffle", action='store_true',  help="not shuffle dataset")
    parser.add_argument("-f", "--four", action='store_true',  help="use eTRIMS-04 dataset")
    parser.add_argument("-n", "--num", metavar="num", type=int, default=5,  help="set number of function")
    parser.add_argument("-p", "--parameter", metavar='file', type=str, help="set trained parameter")
    parser.add_argument("-t", "--tree", metavar='{d,e,b}', default='deb', help="run tree individually")
    
    # ----- etrims_tree -----
    args = parser.parse_args()
    
    t_args = map(lambda x:x in args.tree, ['d','e','b'])
    etrims_tree(radius=args.radius, size=args.size, d_limit=args.limit, unshuffle=args.unshuffle,
                    four=args.four, num=args.num, parameter=args.parameter, t_args=t_args, file_name=args.name)
