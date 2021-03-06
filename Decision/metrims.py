# -*- coding: utf-8 -*-
import os
import sys
import random
import datetime
import numpy as np
import collections
from PIL import Image, ImageFile
from extreme import StackedELMAutoEncoder
import multiprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

##########################################################
##  Decision Tree (for etrims)
##########################################################

class DecisionTree(object):
    def __init__(self, radius=None, num_function=10,
                 condition='gini', seed=123):
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)

    def getNode(self, data, depth):
        return Node(data, self.picture, depth, self.generate_threshold, self.d_limit, self.condition)
            
    def fit(self, picture, d_limit=None):
        #print "Fit"
        # initialize input
        input = []
        self.picture = picture
        self.d_limit = d_limit
        for i,p in enumerate(picture):
            w,h = p.getSize()
            input += [[i,j,k] for j in range(w) for k in range(h)]
            #input += [[i,j,k] for j in range(w) for k in range(h)]

        # fitting with tree_list
        tree_list = [self.getNode(input, 0)]
        for node in tree_list:
            #print "fit"
            node.fit()
            if not node.isTerminal():
                # not terminal
                node.setChildIndex(len(tree_list)) 
                l_data, l_label, r_data, r_label  = node.divide()
                # append l_node and r_node
                depth = node.getDepth() + 1
                tree_list.append(self.getNode(l_data, depth))
                tree_list.append(self.getNode(r_data, depth))

        # set self to tree_list
        self.tree_list = tree_list

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
        count = 0
        while True:
            node = self.tree_list[index]
            node.setPicture(self.picture)
            if node.isTerminal():
                return node.predict(data)
            index = node.predict(data)

            # debug (bellow)
            count += 1
            if count > len(self.tree_list):
                raise Exception('method predict in dt: while loop never ends')

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
        return count * 1.0 / length

    def info(self):
        print "Information"
        print "root node",
        print "node size", len(self.tree_list)
        #print "depth", depth_array
        print "max depth", self.tree_list[-1].getDepth()

        
##########################################################
##  Node (for etrims)
##########################################################

class Node(object):
    def __init__(self, data, picture, depth, gen_threshold, d_limit, condition):
        self.data = data
        self.picture = picture
        self.depth = depth
        self.gen_threshold = gen_threshold
        self.d_limit = d_limit
        self.condition = condition

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
        self.setThreshold(thresholds[index])

    def divide(self, data=None, threshold=None):
        # set data
        if data is None:
            data = self.data
        if not threshold is None:
            self.setThreshold(threshold)
            
        # divide data and label
        lr_data = [[], []]
        lr_label = [[], []]

        # ----- multiprocessing -----
        #print "multiprocessing"
        core = multiprocessing.cpu_count()

        def divide_process(d, index, data_set):
            values = map(self.function, data_set)
            d.update({index:values})

        sub_size = max(len(data) / core, 1)
        sub_data = [data[i:i+sub_size] for i in range(0,len(data),sub_size)]
        manager = multiprocessing.Manager()
        dic = manager.dict()

        jobs = []

        for i,sub in enumerate(sub_data):
            jobs.append(multiprocessing.Process(target=divide_process, args=(dic, i, sub)))

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        #print "done."
        # ----- multiprocessing -----

        i = 0
        for values in dic.values():
            for val in values:
                index = (val > 0)
                element = data[i]
                #print element
                lr_data[index].append(element)
                lr_label[index].append(self.picture[element[0]].getSignal(element[1], element[2]))
                #print lr_label, index, label, i
                i += 1

        l_data, r_data = lr_data
        l_label, r_label = lr_label
        #print self.depth, len(l_data), len(r_data)
        return l_data, l_label, r_data, r_label

    def setThreshold(self, threshold):
        self.selected_dim, self.theta = threshold        


    def function(self, element):
        # set threshold
        #print element, self.selected_dim
        i,  x,  y = element
        dx, dy, c = self.selected_dim
        #print [i, x+dx, y+dy, c],
        #print self.picture[i].getData(x+dx, y+dy)[c]
        try:
            temp = self.picture[i].getData(x+dx, y+dy)[c]
        except IOError:
            #print "IOError"
            temp = self.picture[i].getData(x+dx, y+dy)[c]
        return temp - self.theta

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
        detail = self.label if self.isTerminal() else [self.l_index, self.r_index,
                                                       self.selected_dim, self.theta]
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
    def __init__(self, elm_hidden=None, elm_coef=None,
                 radius=None, num_function=10, condition='gini', seed=123, visualize=False):
        DecisionTree.__init__(self, radius, num_function, condition, seed)
        self.elm_hidden = elm_hidden
        self.elm_coef = elm_coef
        self.visualize = visualize
        self.node_class = ExtremeNode

    def getNode(self, data, depth):
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
    def __init__(self, data, picture, depth, gen_threshold, d_limit, radius, condition):
        Node.__init__(self, data, picture, depth, gen_threshold, d_limit, condition)
        self.radius = radius

    def setThreshold(self, threshold):
        self.selected_dim, self.theta, self.betas, self.biases = threshold
        
    def function(self, element):
        # set threshold            
        i,  x,  y = element
        crop = self.picture[i].cropData(x, y, self.radius)
        for i, beta in enumerate(self.betas):
            bias = self.biases[i]
            crop = sigmoid(np.dot(crop, beta.T) + bias)
        return crop[self.selected_dim] - self.theta
    

    def save(self):
        # return parameter
        detail = self.label if self.isTerminal() else [self.l_index, self.r_index,
                                                       self.selected_dim, self.theta,
                                                       self.betas, self.biases]
        parameter = [self.depth, self.d_limit, self.terminal, detail]
        return parameter

    def load(self, parameter):
        # set parameter
        self.depth, self.d_limit, self.terminal, detail = parameter
        if self.isTerminal():
            self.label = detail
        else:
            self.l_index, self.r_index, self.selected_dim, self.theta, self.betas, self.biases = detail
    
    
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
            return [0,0,0]
        if y < 0 or y >= self.h:
            # out of y_range
            return [0,0,0]
        # in range
        return list(self.data.getpixel((x, y)))

    def getSignal(self, x, y):
        # in range
        return self.signal.getpixel((x, y))

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
    
