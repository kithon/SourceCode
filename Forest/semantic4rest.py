# -*- coding: utf-8 -*-
import os
import random
import datetime
import numpy as np
import collections
from PIL import Image
from extreme import BinaryELMClassifier

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
        
##########################################################
##  ELMTree
##########################################################
class ELMTree(object):
    def __init__(self, radius, sampleSize, numThreshold, numELM, fileName):
        seed = 123
        if radius is None:
            Exception('Error: radius is None.')
        self.radius = radius
        self.sampleSize = sampleSize
        self.numThreshold = numThreshold
        self.np_rng = np.random.RandomState(seed)
        self.fileName = fileName
        self.elm_hidden = numELM
        self.elm_coef = None
        self.dir_name = 'elm/'

    def train(self, train_pic, freq, limit):
        # train decision tree
        print_time('train tree', self.fileName)
        sample = {}
        self.train_pic = train_pic
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w, freq):
                for k in xrange(0, h, freq):
                    # bootstrap
                    if random.random() < freq:
                        sample[i,j,k] = 0
                    
        s_index = 0
        e_index = 1
        node_list = []
        node_length = 1
        currentDepth = 0
        while s_index < e_index:
            for index in xrange(s_index, e_index):
                forceTerminal = not currentDepth < limit
                data = [list(x) for x in sample.iterkeys() if sample[x] == index]
                isTerminal, param = self.getOptParam(data, train_pic, forceTerminal)
                if isTerminal:
                    node_list.append([isTerminal, param, []])
                if not isTerminal:
                    # inner node
                    l_index, r_index = node_length, node_length + 1
                    l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                    for l in l_data:
                        sample[tuple(l)] = l_index
                    for r in r_data:
                        sample[tuple(r)] = r_index
                    node_length += 2
                    node_list.append([isTerminal, param, [l_index, r_index]])
                    
            # update
            currentDepth += 1
            s_index = e_index
            e_index = node_length
            print_time("depth:%d" % (currentDepth), self.fileName)
            
        # initialize input
        print_time('get class distribution', self.fileName)
        input = {}
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w):
                for k in xrange(0, h):
                    input[i,j,k] = 0

        # get class distribution
        for index, node in enumerate(node_list):
            data = [list(x) for x in input.iterkeys() if input[x] == index]
            isTerminal, param, child = node
            if isTerminal:
                # terminal node
                label = self.getLabelList(data, train_pic)
                node_list[index] = [isTerminal, label, []]
            else:
                # inner node
                l_index, r_index = child
                l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                for l in l_data:
                    input[tuple(l)] = l_index
                for r in r_data:
                    input[tuple(r)] = r_index

        # set grown node_list 
        self.node_list = node_list                    

    def test(self, input, test_pic):
        # get class distribution
        index = 0
        while True:
            node = self.node_list[index]
            isTerminal, param, child = node
            if isTerminal:
                # terminal node
                return param
            else:
                # inner node
                val = self.function(input, param, test_pic)
                isLR = child
                index = isLR[val > 0]
                    
    def divide(self, data, param, data_pic):
        lr_data = [[], []]
        lr_label = [[], []]
        for i,element in enumerate(data):
            index = (self.function(element, param, data_pic) > 0)
            lr_data[index].append(element)
            lr_label[index].append(data_pic[element[0]].getSignal(element[1], element[2]))

        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def function(self, element, param, picture):
        weight, bias, beta = param
        i,x,y = element
        crop = picture[i].cropData(x, y, self.radius)
        hidden = sigmoid(np.dot(weight, crop) + bias)
        output = np.dot(hidden, beta) # sigmoid(np.dot(hidden, beta))
        return output - 0.5 # constant theta

    def gini(self, l_label, r_label):
        # get gini (minimize)
        g = 0
        set_size = len(l_label) + len(r_label)
        for label in [l_label, r_label]:
            sub_size = len(label)
            counter = collections.Counter(label).most_common()
            for c in counter:
                p = 1. * c[1] / sub_size
                sub = (1. * sub_size / set_size)
                #print "sub", sub * p * (1. - p) 
                g += sub * p * (1. - p)
        return g

    def getLabelList(self, data, data_pic):
        label_list = []
        for element in data:
            i,x,y = element
            label_list.append(data_pic[i].getSignal(x,y))
        return label_list
    
    def getOptParam(self, data, data_pic, forceTerminal):
        # check terminal 
        label = self.getLabelList(data, data_pic)
        if len(set(label)) == 1 or forceTerminal:
            # terminal
            return True, None

        # find optimized parameter
        obj = None
        optParam = None
        for i in xrange(self.numThreshold):
            param = self.generate_threshold(data)
            l_data, l_label, r_data, r_label = self.divide(data, param, data_pic)
            g = self.gini(l_label, r_label)
            if len(l_data) == 0 or len(r_data) == 0:
                continue
            if obj is None or g < obj:
                optParam = param
                obj = g
        if optParam is None:
            # terminal
            return True, None
        # inner
        return False, optParam                
        
    def generate_threshold(self, numHidden, data):
        # crop data
        sample_input, label = [], []
        num = min(len(data), self.sampleSize)
        sample_index = random.sample(data, num)
        for temp in sample_index:
            i,x,y = temp
            sample_input.append(self.train_pic[i].cropData(x, y, self.radius))
            label.append(self.train_pic[i].getSignal(x,y))

        # label
        label_index = []
        numL, numR = 0, 0
        for l in collections.Counter(label).most_common():
            if numL < numR:
                numL += l[1]
                label_index.append(l[0])
            else:
                numR += l[1]

        sample_signal = [1 if l in label_index else 0 for l in label]
            
        # train elm
        elm = BinaryELMClassifier(n_hidden=numHiden)
        weight, bias, beta = elm.fit(sample_input, sample_signal)
        return weight, bias, beta
            
    def info(self):
        if not self.node_length is None:
            print_time("Information: number of node = %d" % (self.node_length), self.file_name)
        else:
            print_time("Information: self.node_length is not defined", self.file_name)

    

        
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
            temp = []
            for y in xrange(self.h):
                temp.append(list(data.getpixel((x,y))))
            data_list.append(temp)
        self.data = data_list
        
    def setSignal(self, signal):
        signal_list = []
        for x in xrange(self.w):
            temp = []
            for y in xrange(self.h):
                temp.append(signal.getpixel((x,y)))
            signal_list.append(temp)
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
def print_parameter(param, FILE_NAME):
    cmd = 'echo %s >> %s' % (param, FILE_NAME)
    os.system(cmd)
    
def print_time(message, FILE_NAME):
    d = datetime.datetime.today()
    string = '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute,
                                          d.second, d.microsecond, message)
    cmd = 'echo %s >> %s' % (string, FILE_NAME)
    os.system(cmd)

        
##########################################################
##  load_etrims
##########################################################
    
def load_etrims(radius, size, shuffle, name):
    # ----- path initialize -----
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/'
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
    return train_set, test_set

def compute_weight(data_pic):
    # compute label weight from train picture
    label = []
    for i,p in enumerate(data_pic):
        w,h = p.getSize()
        for x in xrange(w):
            for y in xrange(h):
                label.append(data_pic[i].getSignal(x,y))
    label_weight = {}
    for l in collections.Counter(label).most_common():
        label_weight[l[0]] = 1. / l[1]
    return label_weight
        
##########################################################
##  handle forest
##########################################################

def forest_test(forest, test_pic, weight):
    forest_pixel(forest, test_pic, weight)
    
def forest_pixel(forest, test_pic, weight):
    count = 0
    predict = {}
    for i,p in enumerate(test_pic):
        w,h = p.getSize()
        for j in xrange(w):
            for k in xrange(h):
                label = []
                for tree in forest:
                    input = [i,j,k]
                    label += tree.test(input, test_pic)

                hist = collections.Counter(label)
                for h in hist.most_common():
                    label_index = h[0]
                    hist[label_index] *= weight[label_index]
                predict[i,j,k] = hist.most_common()[0][0]
                if predict[i,j,k] == test_pic[i].getSignal(j,k):
                    count += 1
                    
    draw_pixel(predict, test_pic, "forest")
    return 1. * count / len(predict)

def draw_pixel(predict, picture, file_name):
    for i,p in enumerate(picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for j in xrange(w):
            for k in xrange(h):
                image.putpixel((j,k), predict[i,j,k])
        name = file_name + str(i) + ".png"
        image.save(name)

def forest_superpixel(forest, test_pic, weight):
    return

def draw_superpixel(self, predict, file_name):
    for i,p in enumerate(self.test_picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for x in xrange(w):
            for y in xrange(h):
                index = p.getSIndex(x,y)
                image.putpixel((x,y), predict[(i, index)])
        name = file_name + str(i) + ".png"
        image.save(name)

def do_forest(boxSize, dataSize, unShuffle, sampleFreq,
              isELMF
              dataPerTree, depthLimit, numThreshold, numTree,
              numHidden,
              fileName):
    # complement (あとでパラメータとして書き加える)
    sampleSize = boxSize * boxSize * 3
    
    # ----- initialize -----
    print_time('eTRIMS: init', fileName)

    radius = (boxSize - 1) / 2
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize,
                                      shuffle=not unShuffle, name=fileName)

    print_parameter([boxSize, dataSize, unShuffle, sampleFreq], fileName)
    print_parameter([isELMF], fileName)
    print_parameter([dataPerTree, depthLimit, numThreshold, numTree], fileName)
    print_parameter([numHidden], fileName)
    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d'
               % (radius, str(depthLimit), dataSize, numThreshold), fileName)

    # compute label weight
    weight = compute_weight(train_pic)
    
    if isELMF:
        print_time('ELM forest', fileName)
        print_time('init', fileName)
        forest = []
        print_time('train', fileName)
        for i in xrange(numTree):
            print_time('tree: %i' % i, fileName)
            tree = ELMTree(radius=radius, sampleSize=sampleSize, numThreshold=numThreshold,
                           numELM = numHiden, fileName=fileName)        
            tree.train(train=train_pic, freq=sampleFreq, limit=depthLimit)
            forest.append(tree)

        print_time('test', fileName)
        forest_test(forest, test_pic, weight)


    # ----- finish -----
    print_time('eTRIMS: finish', fileName)

    
