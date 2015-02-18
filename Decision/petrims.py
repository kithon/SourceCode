# -*- coding: utf-8 -*-
import os
import random
import datetime
import numpy as np
import collections
import argparse
import multiprocessing
import linecache
from ast import literal_eval
from PIL import Image
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
                 'picture', 'sig_picture', 'node_length', 'parameter_list']
    def __init__(self, radius=None, num_function=10, condition='gini', seed=123, file_name=None):
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)
        self.dir_name = 'decision/'
        self.file_name = file_name

    def getNode(self, data=None, signal=None, depth=None):
        return Node(data, self.picture, signal, self.sig_picture, depth, self.generate_threshold, self.d_limit, self.condition)
            
    def fit(self, picture, sig_picture, d_limit=None, overlap=True):
        # ----- initialize -----
        # -*- input, picture, param, d_limit -*-
        self.picture = picture
        self.sig_picture = sig_picture
        self.d_limit = d_limit
        input = {}
        for i,p in enumerate(picture):
            w,h = p.getSize()
            if overlap:
                input = {(i,j,k):0 for j in xrange(w) for k in xrange(h)}
            else:
                input = {(i,j,k):0 for j in xrange(self.radius, w, 2*self.radius+1) for k in xrange(self.radius, h, 2*self.radius+1)}

        count = 0
        fix_count = 0
        signal = {}
        for i,p in enumerate(sig_picture):
            w,h = p.getSize()
            signal = {(i,j,k):0 for j in xrange(w) for k in xrange(h)}
        length = len(signal)

        # -*- execution_list, wait_list, node_length -*-
        node_length = 1
        core = multiprocessing.cpu_count()

        # -*- current_depth, index -*-
        current_depth = 0
        s_index = 0
        e_index = 1
        while s_index < e_index:
            count = 0

            # -*- initialize jobs, dic -*-
            jobs = []
            dic = multiprocessing.Manager().dict()
            num_process = min(core, e_index - s_index)

            # -*- distribute exec_list -*-
            for i in xrange(num_process):
                index_list = range(i + s_index, e_index, num_process)
                input_list = [[list(x) for x in input.iterkeys() if input[x] == index] for index in index_list]
                node_list = [self.getNode(data=temp, depth=current_depth) for temp in input_list]
                jobs.append(multiprocessing.Process(target=fit_process, args=(dic, index_list, node_list)))

            # -*- multiprocessing -*-
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
                            
            for index in xrange(s_index, e_index):
                input_list = [list(x) for x in input.iterkeys() if input[x] == index]
                signal_list = [list(x) for x in signal.iterkeys() if signal[x] == index]
                node = self.getNode(input_list, signal_list, current_depth)
                parameter = dic.get(index)
                node.load(parameter)
                point = node.getScore()
                if not node.isTerminal():
                    count += point
                    node.setChildIndex(node_length)
                    l_data, l_label, r_data, r_label = node.divide()
                    l_sign, l_pred,  r_sign, r_pred  = node.divide(data=signal_list, picture=self.sig_picture)
                    l_index, r_index = node.getChildIndex()
                    for l in l_data:
                        input[tuple(l)] = l_index
                    for r in r_data:
                        input[tuple(r)] = r_index
                    for l in l_sign:
                        signal[tuple(l)] = l_index
                    for r in r_sign:
                        signal[tuple(r)] = r_index
                    node_length += 2
                else:
                    fix_count += point            

            count += fix_count
            score = count * 1.0 / length
            print_time("depth:%d score = %f" % (current_depth, score), self.file_name)
                    
            # -*- update current_depth, index -*-
            current_depth += 1
            s_index = e_index
            e_index = node_length

        # -*- set node_length -*-
        self.node_length = node_length

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
            
    def info(self):
        if not self.node_length is None:
            print_time("Information: number of node = %d" % (self.node_length), self.file_name)
        else:
            print_time("Information: self.node_length is not defined", self.file_name)

        
##########################################################
##  Node (for etrims)
##########################################################

class Node(object):
    __slots__ = ['data', 'picture', 'signal', 'sig_picture','depth', 'gen_threshold',
                 'd_limit', 'condition', 'l_index', 'r_index',
                 'terminal', 'label', 'selected_dim', 'theta']
    def __init__(self, data=None, picture=None, signal=None, sig_picture=None, depth=None, gen_threshold=None, d_limit=None, condition=None):
        if not data is None:
            self.data = data
            self.picture = picture
            self.signal = signal
            self.sig_picture = sig_picture
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
            count = 0
            limit = 50
            while len(l_data) == 0 or len(r_data) == 0:
                thresholds = [t for t in self.gen_threshold(self.data)]
                self.opt_threshold(self.data, thresholds)
                l_data, l_label, r_data, r_label = self.divide()
                if limit < count:
                    self.terminal = True
                    return                    
                
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

    def divide(self, data=None, threshold=None, picture=None):
        # set data
        if data is None:
            data = self.data
        if picture is None:
            picture = self.picture

        # divide data and label
        lr_data = [[], []]
        lr_label = [[], []]
        for i, element in enumerate(data):
            #print element
            index = (self.function(element, threshold) > 0)
            lr_data[index].append(element)
            lr_label[index].append(picture[element[0]].getSignal(element[1], element[2]))
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
        for i, element in enumerate(self.signal):
            i,x,y = element
            if self.sig_picture[i].getSignal(x,y) == self.label:
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

    def getChildIndex(self):
        # get child index
        return self.l_index, self.r_index

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
                 'picture', 'sig_picture', 'node_length', 'parameter_list',
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
        

    def getNode(self, data=None, signal=None, depth=None):
        return ExtremeNode(data, self.picture, signal, self.sig_picture, depth, self.generate_threshold, self.d_limit, self.radius, self.condition)
    
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
    __slots__ = ['data', 'picture', 'signal', 'sig_picture', 'depth', 'gen_threshold',
                 'd_limit', 'radius', 'condition', 'l_index', 'r_index',
                 'terminal', 'label', 'selected_dim', 'theta',
                 'betas', 'biases']
    def __init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, radius, condition):
        Node.__init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, condition)
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
##  Random ExtremeDecision Tree (for etrims)
##########################################################

class RandomExtremeDecisionTree(DecisionTree):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'dir_name', 'file_name',
                 'picture', 'sig_picture', 'node_length', 'parameter_list',
                 'n_input', 'n_hidden', 'visualize']
    def __init__(self, n_input=None, n_hidden=None, radius=1, num_function=10, condition='gini', seed=123, visualize=False, file_name=None):
        DecisionTree.__init__(self, radius, num_function, condition, seed, file_name)
        if n_input is None:
            n_input = 3 * (2*radius+1) * (2*radius+1)
        if n_hidden is None:
            n_hidden = 2 * n_input
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.visualize = visualize
        self.dir_name = 'random/'        

    def getNode(self, data=None, signal=None, depth=None):
        return RandomExtremeNode(data, self.picture, signal, self.sig_picture, depth, self.generate_threshold, self.d_limit, self.radius, self.condition)
    
    def generate_threshold(self, data):
        for i in xrange(self.num_function):            
            weight = self.np_rng.uniform(low = 0.0, high = 1.0, size = (self.n_input, self.n_hidden))
            bias = self.np_rng.uniform(low = 0.0, high = 1.0, size = self.n_hidden)

            # regularize weight
            for i,w in enumerate(weight):
                denom = np.linalg.norm(w)
                if denom != 0:
                    weight[i] = w / denom

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                denom = bias / denom

            sample = []
            num = min(len(data), (2*self.radius+1)*(2*self.radius+1))
            sample_index = random.sample(data, num)
            for temp in sample_index:
                i,x,y = temp
                sample.append(self.picture[i].cropData(x, y, self.radius))

            numpy_data = np.array(sigmoid(np.dot(weight.T, sample) + bias))
            selected_dim = self.np_rng.randint(self.n_hidden)
            selected_row = numpy_data.T[selected_dim]
            min_row = selected_row.min()
            max_row = selected_row.max()
            theta = self.np_rng.rand() * (max_row - min_row) + min_row
            yield selected_dim, theta, weight, bias

            
##########################################################
##  Random Extreme Node (for etrims)
##########################################################

class RandomExtremeNode(Node):
    __slots__ = ['data', 'picture', 'signal', 'sig_picture', 'depth', 'gen_threshold',
                 'd_limit', 'radius', 'condition', 'l_index', 'r_index',
                 'terminal', 'label', 'selected_dim', 'theta',
                 'weights', 'biases']
    def __init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, radius, condition):
        Node.__init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, condition)
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
        self.selected_dim, self.theta, self.weights, self.biases = thresholds[index]
        
    def function(self, element, threshold=None):
        # set threshold
        if threshold is None:
            selected_dim = self.selected_dim
            theta = self.theta
            weights = self.weights
            biases = self.biases
        else:
            selected_dim, theta, weights, biases = threshold
            
        i,  x,  y = element
        crop = self.picture[i].cropData(x, y, self.radius)
        for i, weight in enumerate(weights):
            bias = biases[i]
            crop = sigmoid(np.dot(weight.T, crop) + bias)
        return crop[selected_dim] - theta
    

    def save(self):
        # return parameter
        detail = [] if self.isTerminal() else [self.l_index, self.r_index,
                                               self.selected_dim, self.theta,
                                               map(lambda n:n.tolist(), self.weights),
                                               map(lambda n:n.tolist(), self.biases)]
        parameter = [self.depth, self.d_limit, self.terminal, self.label, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, self.label, detail = parameter
        if not self.isTerminal():
            l, r, s, t, w, bi = detail
            self.l_index = l
            self.r_index = r
            self.selected_dim = s
            self.theta = t
            self.weights = map(np.array, w)
            self.biases = map(np.array, bi)
    
##########################################################
##  ExtremeBinaryDecision Tree (for etrims)
##########################################################

class BinaryExtremeDecisionTree(DecisionTree):
    __slots__ = ['radius', 'num_function', 'condition',
                 'np_rng', 'd_limit', 'dir_name', 'file_name',
                 'picture', 'sig_picture', 'node_length', 'parameter_list',
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

    def getNode(self, data=None, signal=None, depth=None):
        return BinaryExtremeNode(data, self.picture, signal, self.sig_picture, depth, self.generate_threshold, self.d_limit, self.radius, self.condition)
    
    def generate_threshold(self, data):
        #print "Generate ", size, " divide functions"
        for i in xrange(self.num_function):            
            elm = BinaryELMClassifier(n_hidden=self.elm_hidden, coef=self.elm_coef)
            sample_input, sample_label  = [], []
            num = min(len(data), (2*self.radius+1)*(2*self.radius+1))

            label = set()
            for d in data:
                i,x,y = d
                label.add(self.picture[i].getSignal(x, y))
            sig = random.sample(label, 2)
            if len(label) < 2:
                raise Exception('BEDT:generate_threshold')
            
            l_data, r_data = [], []
            for d in data:
                i,x,y = d
                if sig[0] == self.picture[i].getSignal(x, y):
                    l_data.append(d)
                elif sig[1] == self.picture[i].getSignal(x, y):
                    r_data.append(d)

            l_data = random.sample(l_data, min(num/2, len(l_data)))
            for l in l_data:
                i,x,y = l
                sample_input.append(self.picture[i].cropData(x, y, self.radius))
                sample_label.append(0)

            r_data = random.sample(r_data, min(num/2, len(r_data)))
            for r in r_data:
                i,x,y = r
                sample_input.append(self.picture[i].cropData(x, y, self.radius))
                sample_label.append(0)
                
            weight, bias, beta = elm.fit(sample_input, sample_label)
            yield weight, bias, beta


##########################################################
##  Binary Extreme Node (for etrims)
##########################################################

class BinaryExtremeNode(Node):
    __slots__ = ['data', 'picture', 'signal', 'sig_picture', 'depth', 'gen_threshold',
                 'd_limit', 'radius', 'condition', 'l_index', 'r_index',
                 'terminal', 'label',
                 'weight', 'bias', 'beta']
    def __init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, radius, condition):
        Node.__init__(self, data, picture, signal, sig_picture, depth, gen_threshold, d_limit, condition)
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
        data_list = {}
        for x in xrange(self.w):
            for y in xrange(self.h):
                data_list[x,y] = list(data.getpixel((x,y)))
        self.data = data_list
        
    def setSignal(self, signal):
        signal_list = {}
        for x in xrange(self.w):
            for y in xrange(self.h):
                signal_list[x,y] = signal.getpixel((x,y))
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
        return self.data[x,y]

    def getSignal(self, x, y):
        # in range
        return self.signal[x,y]

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
    
def load_etrims(radius, size, is08, shuffle, name, t_index=None):
    # ----- make up -----
    isInit = t_index is None
    
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
    if isInit:
        train_index = random.sample(range(DATA_SIZE), TRAIN_SIZE) if shuffle else range(TRAIN_SIZE)
    else:
        train_index = t_index

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
        if isInit:
            print_time("eTRIMS: %s" % file_name, name)

    # ----- finish -----
    if isInit:
        print_parameter(train_index, name)
        print_time("eTRIMS: train=%d test=%d" % (len(train_set), len(test_set)), name)
    return train_set, test_set


def etrims_tree(radius, size, d_limit, unshuffle, cram, four, num, parameter, t_args, file_name):
    # ----- initialize -----
    print_parameter([radius, size, d_limit, unshuffle, four, num, t_args], file_name)
    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d' % (radius, str(d_limit), size, num), file_name)
    print_time('eTRIMS: load', file_name)
    train_set, test_set = load_etrims(radius=radius, size=size, is08=not four, shuffle=not unshuffle, name=file_name)
    isDT, isEDT, isREDT, isBEDT = t_args
    
    # ----- Decision Tree -----
    if isDT:        
        print_time('DecisionTree: init', file_name)
        dt = DecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('DecisionTree: train', file_name)
        dt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)
        
        print_time('DecisionTree: info', file_name)
        dt.info()
    

    # ----- Extreme Decision Tree -----
    if isEDT:
        print_time('ExtremeDecisionTree: init', file_name)
        edt = ExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('ExtremeDecisionTree: train', file_name)
        edt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)

        print_time('ExtremeDecisionTree: info', file_name)
        edt.info()

    # ----- Random Extreme Decision Tree -----
    if isREDT:
        print_time('RandomExtremeDecisionTree: init', file_name)
        redt = RandomExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('RandomExtremeDecisionTree: train', file_name)
        redt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)

        print_time('RandomExtremeDecisionTree: info', file_name)
        redt.info()

    # ----- Binary Extreme Decision Tree -----
    if isBEDT:
        print_time('BinaryExtremeDecisionTree: init', file_name)
        bedt = BinaryExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('BinaryExtremeDecisionTree: train', file_name)
        bedt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)
        
        print_time('BinaryExtremeDecisionTree: info', file_name)
        bedt.info()


    # ----- finish -----
    print_time('eTRIMS: finish', file_name)


def etrims_tree_makeup(radius, size, d_limit, unshuffle, cram, four, num, parameter, t_args, file_name):
    # ----- initialize -----
    print_time('eTRIMS: make up')
    param = literal_eval(linecache(parameter, 1))
    radius, size, d_limit, unshuffle, four, num = param[0:6]
    t_index = literal_eval(linecache(parameter, 3+size))
    train_set, test_set = load_etrims(radius=radius, size=size, is08=not four, shuffle=not unshuffle, name=file_name, t_index=t_index)
    isDT, isEDT, isREDT, isBEDT = t_args
    
    # ----- Decision Tree -----
    if isDT:        
        print_time('DecisionTree: init', file_name)
        dt = DecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('DecisionTree: train', file_name)
        dt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)
        
        print_time('DecisionTree: info', file_name)
        dt.info()
    

    # ----- Extreme Decision Tree -----
    if isEDT:
        print_time('ExtremeDecisionTree: init', file_name)
        edt = ExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('ExtremeDecisionTree: train', file_name)
        edt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)

        print_time('ExtremeDecisionTree: info', file_name)
        edt.info()

    # ----- Random Extreme Decision Tree -----
    if isREDT:
        print_time('RandomExtremeDecisionTree: init', file_name)
        redt = RandomExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('RandomExtremeDecisionTree: train', file_name)
        redt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)

        print_time('RandomExtremeDecisionTree: info', file_name)
        redt.info()
        
    # ----- Binary Extreme Decision Tree -----
    if isBEDT:
        print_time('BinaryExtremeDecisionTree: init', file_name)
        bedt = BinaryExtremeDecisionTree(radius=radius, num_function=num, file_name=file_name)
        
        print_time('BinaryExtremeDecisionTree: train', file_name)
        bedt.fit(train_set, test_set, d_limit=d_limit, overlap=cram)
        
        print_time('BinaryExtremeDecisionTree: info', file_name)
        bedt.info()


    # ----- finish -----
    print_time('eTRIMS: finish', file_name)

    

if __name__ == '__main__':
    # ----- parser description -----
    parser = argparse.ArgumentParser(description='Test eTRIMS-08 Segmentation Dataset (need etrims_tree.py)')
    parser.add_argument("name", type=str, default='result.log', help="set file name")
    parser.add_argument("radius", type=int, default=5, nargs='?', help="set image radius")
    parser.add_argument("size", type=int, default=60, nargs='?', help="set data size")
    parser.add_argument("limit", type=int, nargs='?', help="set depth limit")
    parser.add_argument("-u", "--unshuffle", action='store_true',  help="not shuffle dataset")
    parser.add_argument("-c", "--cram", action='store_false',  help="not overlap")
    parser.add_argument("-f", "--four", action='store_true',  help="use eTRIMS-04 dataset")
    parser.add_argument("-n", "--num", metavar="num", type=int, default=5,  help="set number of function")
    parser.add_argument("-p", "--parameter", metavar='file', type=str, help="set trained parameter")
    parser.add_argument("-s", "--seed", type=int, default=1, help="seed")
    parser.add_argument("-t", "--tree", metavar='{d,e,r,b}', default='derb', help="run tree individually")
    
    # ----- etrims_tree -----
    args = parser.parse_args()
    
    t_args = map(lambda x:x in args.tree, ['d','e','r','b'])
    if args.parameter is None:
        etrims_tree(radius=args.radius, size=args.size, d_limit=args.limit, unshuffle=args.unshuffle, cram=args.cram,
                    four=args.four, num=args.num, parameter=args.parameter, t_args=t_args, file_name=args.name)
    else:
        etrims_tree_makeup(radius=args.radius, size=args.size, d_limit=args.limit, unshuffle=args.unshuffle, cram=args.cram,
                           four=args.four, num=args.num, parameter=args.parameter, t_args=t_args, file_name=args.name)
   
