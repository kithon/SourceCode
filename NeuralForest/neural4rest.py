# -*- coding: utf-8 -*-
import os
import h5py
import random
import datetime
import operator
import slic as sc
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
    def __init__(self, radius, sampleSize, numThreshold, numELM, weight, fileName):
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
        self.weight = weight
        self.tree_name = 'elm_tree.h5'
        self.dir_name = 'elm_'

    def fit(self, train_pic, test_pic, freq, limit, num):
        check_depth = 10

        sample = {}
        self.train_pic = train_pic
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w, freq):
                for k in xrange(0, h, freq):
                    # bootstrap
                    if random.random() < freq:
                        sample[i,j,k] = 0

        train_input = {}
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w):
                for k in xrange(0, h):
                    train_input[i,j,k] = 0

        test_input = {}
        for i in xrange(len(test_pic)):
            w,h = test_pic[i].getSize()
            for j in xrange(w):
                for k in xrange(h):
                    test_input[i,j,k] = 0
                    
        s_index = 0
        e_index = 1
        node_length = 1
        currentDepth = 0
        predict_hist = {}
        while s_index < e_index and (currentDepth < limit and not limit is None):
            print_time("depth:%d" % (currentDepth), self.fileName)
            currentDepth += 1
            print_time("num of node:%d" % (e_index - s_index), self.fileName)
            for index in xrange(s_index, e_index):
                forceTerminal = not currentDepth < limit
                sample_data = [list(x) for x in sample.iterkeys() if sample[x] == index]
                train_data = [list(x) for x in train_input.iterkeys() if train_input[x] == index]
                test_data = [list(x) for x in test_input.iterkeys() if test_input[x] == index]

                isTerminal, param = self.getOptParam(sample_data, train_pic, forceTerminal)
                if isTerminal:
                    hist = self.getHist(train_data, train_pic)
                    for d in test_data:
                        predict_hist[tuple(d)] = hist
                if not isTerminal:
                    # sample divide
                    l_index, r_index = node_length, node_length + 1
                    l_data, l_label, r_data, r_label = self.divide(sample_data, param, train_pic)
                    for l in l_data:
                        sample[tuple(l)] = l_index
                    for r in r_data:
                        sample[tuple(r)] = r_index

                    # train divide
                    l_data, l_label, r_data, r_label = self.divide(train_data, param, train_pic)
                    for l in l_data:
                        train_input[tuple(l)] = l_index
                    for r in r_data:
                        train_input[tuple(r)] = r_index

                    # test divide
                    l_data, l_label, r_data, r_label = self.divide(test_data, param, test_pic)
                    for l in l_data:
                        test_input[tuple(l)] = l_index
                    for r in r_data:
                        test_input[tuple(r)] = r_index

                    # add node_length
                    node_length += 2

                    if currentDepth % check_depth == 0 or (s_index < e_index and (currentDepth < limit and not limit is None)):
                        hist = self.getHist(train_data, train_pic)
                        for d in test_data:
                            predict_hist[tuple(d)] = hist
                            
                if currentDepth % check_depth == 0 or (s_index < e_index and (currentDepth < limit and not limit is None)):
                    # print score and draw picture
                    # pixel level
                    print_time("print score and draw picture", self.fileName)
                    predict, score = predict_pixel(predict_hist, test_pic, self.fileName)
                    Global, Accuracy, Class_Avg, Jaccard = score
                    print_time('tree%d_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (num, Global, Accuracy, Class_Avg, Jaccard), self.fileName)
                    draw_pixel(predict, test_pic, self.dir_name + "tree%d_depth%d_pixel" % (num, currentDepth))

                    # super-pixel level
                    predict, score = predict_superpixel(predict_hist, test_pic, self.fileName)
                    Global, Accuracy, Class_Avg, Jaccard = score
                    print_time('tree%d_superpixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (num, Global, Accuracy, Class_Avg, Jaccard), self.fileName)
                    draw_superpixel(predict, test_pic, self.dir_name + "tree%d_depth%d_superpixel" % (num, currentDepth))
                    
            # update index
            s_index = e_index
            e_index = node_length

        self.node_length = node_length
        print_time("node length:%d" % self.node_length, self.fileName)

        
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
        node_length = 1
        currentDepth = 0
        self.node_name = 'node_'
        h5file = h5py.File(self.tree_name, 'w') # or 'a'
        while s_index < e_index and (currentDepth < limit and not limit is None):
            print_time("depth:%d" % (currentDepth), self.fileName)
            currentDepth += 1
            print_time("num of node:%d" % (e_index - s_index), self.fileName)
            for index in xrange(s_index, e_index):
                dir = self.node_name + str(index)
                h5file.create_group(dir)
                forceTerminal = not currentDepth < limit
                data = [list(x) for x in sample.iterkeys() if sample[x] == index]
                isTerminal, param = self.getOptParam(data, train_pic, forceTerminal)
                if isTerminal:
                    h5file.create_dataset(dir + '/isTerminal', data = isTerminal)
                if not isTerminal:
                    # inner node
                    l_index, r_index = node_length, node_length + 1
                    l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                    for l in l_data:
                        sample[tuple(l)] = l_index
                    for r in r_data:
                        sample[tuple(r)] = r_index
                    node_length += 2
                    h5file.create_dataset(dir + '/isTerminal', data = isTerminal)
                    # ---------- param ----------
                    weight, bias, beta = param
                    h5file.create_dataset(dir + '/weight', data = weight)
                    h5file.create_dataset(dir + '/bias', data = bias)
                    h5file.create_dataset(dir + '/beta', data = beta)
                    # ---------- param ----------
                    h5file.create_dataset(dir + '/child', data = [l_index, r_index])
                    
            # update
            s_index = e_index
            e_index = node_length
        #h5file.flush()
        #h5file.close()
            
        # initialize input
        print_time('get class distribution', self.fileName)
        input = {}
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for j in xrange(0, w):
                for k in xrange(0, h):
                    input[i,j,k] = 0

        # get class distribution
        #h5file = h5py.File(self.tree_name, 'a') 
        for index in xrange(node_length):
            dir = self.node_name + str(index)
            data = [list(x) for x in input.iterkeys() if input[x] == index]
            isTerminal = h5file[dir + '/isTerminal'].value
            if isTerminal:
                # terminal node
                hist = self.getHist(data, train_pic)
                list_hist = [hist[i] for i in xrange(1,9)]
                h5file.create_dataset(dir + '/hist' , data = list_hist)
            else:
                # inner node
                # ---------- param ----------
                weight = h5file[dir + '/weight'].value
                bias = h5file[dir + '/bias'].value
                beta = h5file[dir + '/beta'].value
                param = weight, bias, beta
                # ---------- param ----------
                l_index, r_index = h5file[dir + '/child'].value
                l_data, l_label, r_data, r_label = self.divide(data, param, train_pic)
                for l in l_data:
                    input[tuple(l)] = l_index
                for r in r_data:
                    input[tuple(r)] = r_index

        # set grown node_list 
        self.node_length = node_length
        h5file.flush()
        h5file.close()

    def test(self, test_pic):
        # initialize input
        input = {}
        for i in xrange(len(test_pic)):
            w,h = test_pic[i].getSize()
            for j in xrange(w):
                for k in xrange(h):
                    input[i,j,k] = 0

        # predict class distribution
        predict = {}
        h5file = h5py.File(self.tree_name, 'r')
        for index in xrange(self.node_length): 
            dir = self.node_name + str(index)
            data = [list(x) for x in input.iterkeys() if input[x] == index]
            isTerminal = h5file[dir + '/isTerminal'].value
            if isTerminal:
                # terminal node
                list_hist = h5file[dir + '/hist'].value
                hist = {i:list_hist[i-1] for i in xrange(1,9)}
                for d in data:
                    predict[tuple(d)] = hist
            else:
                # inner node
                # ---------- param ----------
                weight = h5file[dir + '/weight'].value
                bias = h5file[dir + '/bias'].value
                beta = h5file[dir + '/beta'].value
                param = weight, bias, beta
                # ---------- param ----------
                l_index, r_index = h5file[dir + '/child'].value
                l_data, l_label, r_data, r_label = self.divide(data, param, test_pic)
                for l in l_data:
                    input[tuple(l)] = l_index
                for r in r_data:
                    input[tuple(r)] = r_index
        h5file.flush()
        h5file.close()
        return predict

                    
    def divide(self, data, param, data_pic):
        lr_data = [[], []]
        lr_label = [[], []]

        lr_list = map(lambda element:self.function(element, param, data_pic) > 0, data)
        sig_list = map(lambda element:data_pic[element[0]].getSignal(element[1], element[2]), data)
        for i,(lr, sig) in enumerate(zip(lr_list, sig_list)):
            lr_data[lr].append(data[i])
            lr_label[lr].append(sig)

        l_data, r_data = lr_data
        l_label, r_label = lr_label
        return l_data, l_label, r_data, r_label

    def function(self, element, param, picture):
        i,x,y = element
        weight, bias, beta = param
        crop = picture[i].cropData(x, y, self.radius)
        hidden = sigmoid(np.dot(weight.T, crop) + bias)
        output = np.dot(beta.T, hidden) # sigmoid(np.dot(hidden, beta))
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

    def getHist(self, data, data_pic):
        label_list = self.getLabelList(data, data_pic)
        hist = collections.Counter(label_list)
        for h in hist.most_common():
            label_index = h[0]
            hist[label_index] *= self.weight[label_index]        
        return {i:hist[i] for i in xrange(1,9)} # for Etrims8
    
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
            #print_time('th: %i' % i, self.fileName)
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
        
    def generate_threshold(self, data):
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
        elm = BinaryELMClassifier(n_hidden=self.elm_hidden)
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
    __slots__ = ['data', 'signal', 'spixel', 'palette',
                 'slength', 'scenter', 'sdic', 'w', 'h']
    def __init__(self, data, signal, spixel):
        self.w, self.h = data.size
        self.palette = signal.getpalette()
        self.setData(data)
        self.setSignal(signal)
        self.setSpixel(spixel)

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

    def setSpixel(self, spixel):
        self.slength = np.max(spixel) + 1
        self.spixel = spixel.T.tolist()

        super_dic = {}
        super_count = np.zeros(self.slength)
        super_label = []
        for i in xrange(self.slength):
            super_label.append([])
        
        super_center = np.zeros((self.slength, 2))
        for x in xrange(self.w):
            for y in xrange(self.h):
                super_center[self.spixel[x][y]] += [x, y]
                super_count[self.spixel[x][y]] += 1
                super_label[self.spixel[x][y]].append(self.getSignal(x,y))
        for i,c in enumerate(super_count):
            super_center[i] /= c
            
        self.scenter = super_center.astype(np.int64).tolist()
        for i in xrange(self.slength):
            super_dic[i] = collections.Counter(super_label[i]).most_common()[0][0]
        self.sdic = super_dic
        
    def getSize(self):
        return self.w, self.h

    def getSSize(self):
        return self.slength

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

    def getPalette(self):
        return self.palette        
    
    def getSIndex(self, x, y):
        return self.spixel[x][y]

    def getSData(self, index, dx, dy):
        x,y = self.scenter[index]
        x,y = x+dx, y+dy
        if x < 0 or x >= self.w:
            # out of x_range
            return [0,0,0]
        if y < 0 or y >= self.h:
            # out of y_range
            return [0,0,0]
        # in range
        return self.data[x][y]

    def getSSignal(self, index):
        return self.sdic[index]
    
    def cropData(self, x, y, radius):
        crop = []
        for dx in range(x-radius, x+radius+1):
            for dy in range(y-radius, y+radius+1):
                crop += self.getData(dx, dy)
        crop = (1. * np.array(crop) / 255).tolist()
        return crop

    def cropSData(self, index, radius):
        x,y = self.scenter[index]
        return self.cropData(x,y,radius)
    
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
    
def load_etrims(radius, size, shuffle, name, n_superpixels, compactness):
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
        spixel = sc.slic_n(np.array(image), n_superpixels, compactness)

        
        # get index and set picture
        index = i in train_index
        picture = Pic(image, annotation, spixel)
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

def forest_test(forest, test_pic, fileName, dirName = ""):
    # ---------- get class distribution (pixel-wise) ----------
    hist = {}
    hist_list = []
    for i,tree in enumerate(forest):
        print_time('tree:%d' % i, fileName)
        hist_list.append(tree.test(test_pic))

    print_time('predict', fileName)       
    for i,p in enumerate(test_pic):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                hist[i,j,k] = {}
                for c in xrange(1,9):
                    hist[i,j,k][c] = sum(map(lambda x:x[c], map(lambda h:h[i,j,k], hist_list)))

    # ---------- pixel wise ----------
    # """
    # -*- get predict and score -*-
    predict_list, score_list = [], []
    predict, score = predict_pixel(hist, test_pic, fileName)
    for h in hist_list:
        p,s = predict_pixel(h, test_pic, fileName)
        predict_list.append(p)
        score_list.append(s)

    Global, Accuracy, Class_Avg, Jaccard = score
    print_time('forest_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (Global, Accuracy, Class_Avg, Jaccard), fileName)
    for i,s in enumerate(score_list):
        Global, Accuracy, Class_Avg, Jaccard = s
        print_time('tree%d_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (i, Global, Accuracy, Class_Avg, Jaccard), fileName)
        
    print_time('draw_pixel', fileName)    
    draw_pixel(predict, test_pic, dirName + "forest_pixel")
    for i,p in enumerate(predict_list):
        draw_pixel(p, test_pic, dirName + 'tree%d_pixel' % i)
    # """

    # ---------- super-pixel wise ----------
    # """
    # -*- get predict and score -*-
    predict_list, score_list = [], []
    predict, score = predict_superpixel(hist, test_pic, fileName)
    for h in hist_list:
        p,s = predict_superpixel(h, test_pic, fileName)
        predict_list.append(p)
        score_list.append(s)

    Global, Accuracy, Class_Avg, Jaccard = score
    print_time('forest_super: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (Global, Accuracy, Class_Avg, Jaccard), fileName)
    for i,s in enumerate(score_list):
        Global, Accuracy, Class_Avg, Jaccard = s
        print_time('tree%d_super: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (i, Global, Accuracy, Class_Avg, Jaccard), fileName)
        
    print_time('draw_super', fileName)    
    draw_superpixel(predict, test_pic, dirName + "forest_super")
    for i,p in enumerate(predict_list):
        draw_superpixel(p, test_pic, dirName + 'tree%d_super' % i)
    # """

        
def predict_pixel(hist, picture, fileName):
    # ---------- pixel wise ----------q
    TP, TN, FP, FN = 0, 0, 0, 0
    one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0
    predict = {}
    for i,p in enumerate(picture):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                label = picture[i].getSignal(j,k)
                # predict & count
                predict[i,j,k] = max(hist[i,j,k].iteritems(), key=operator.itemgetter(1))[0]
                if predict[i,j,k] == label:
                    one_TP += 1
                    one_TN += 7
                else:
                    one_FP += 7
                    one_FN += 1
        Global = 1. * one_TP / (width * height)
        Accuracy = 1. * (one_TP + one_TN) / (one_TP + one_TN + one_FP + one_FN)
        Class_Avg = 1. * one_TP / (one_TP + one_FN)
        Jaccard = 1. * one_TP / (one_TP + one_FP + one_FN)
        print_time('%dth picture: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (i, Global, Accuracy, Class_Avg, Jaccard), fileName)
        TP += one_TP
        TN += one_TN
        FP += one_FP
        FN += one_FN
        one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0

    length = len(predict)
    Global = 1. * TP / length
    Accuracy = 1. * (TP + TN) / (TP + TN + FP + FN)
    Class_Avg = 1. * TP / (TP + FN)
    Jaccard = 1. * TP / (TP + FP + FN)
    return predict, [Global, Accuracy, Class_Avg, Jaccard]

def draw_pixel(predict, picture, file_name):
    # ---------- pixel wise ----------
    for i,p in enumerate(picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for j in xrange(w):
            for k in xrange(h):
                image.putpixel((j,k), predict[i,j,k])
        name = file_name + str(i) + ".png"
        image.save(name)
    
def predict_superpixel(hist, picture, fileName):
    # ---------- super-pixel wise ----------
    # compress hist
    length = 0
    super_hist = {}
    for i,p in enumerate(picture):
        width, height = p.getSize()
        length += (width * height)
        for index in xrange(p.getSSize()):
            super_hist[i,index] = {col:0 for col in xrange(1,9)}         
        for j in xrange(width):
            for k in xrange(height):
                index = p.getSIndex(j,k)
                for col in xrange(1,9):
                    super_hist[i,index][col] = super_hist[i,index][col] + hist[i,j,k][col] 

    # predict
    predict = {}
    for i,p in enumerate(picture):
        for index in xrange(p.getSSize()):
            predict[i,index] = max(super_hist[i,index].iteritems(), key=operator.itemgetter(1))[0]

    # count
    TP, TN, FP, FN = 0, 0, 0, 0
    one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0
    for i,p in enumerate(picture):
        width, height = p.getSize()
        for j in xrange(width):
            for k in xrange(height):
                label = p.getSignal(j,k)
                index = p.getSIndex(j,k)
                if predict[i,index] == label:
                    one_TP += 1
                    one_TN += 7
                else:
                    one_FP += 7
                    one_FN += 1                    
        Global = 1. * one_TP / (width * height)
        Accuracy = 1. * (one_TP + one_TN) / (one_TP + one_TN + one_FP + one_FN)
        Class_Avg = 1. * one_TP / (one_TP + one_FN)
        Jaccard = 1. * one_TP / (one_TP + one_FP + one_FN)
        print_time('%dth picture: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (i, Global, Accuracy, Class_Avg, Jaccard), fileName)
        TP += one_TP
        TN += one_TN
        FP += one_FP
        FN += one_FN
        one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0

    Global = 1. * TP / length
    Accuracy = 1. * (TP + TN) / (TP + TN + FP + FN)
    Class_Avg = 1. * TP / (TP + FN)
    Jaccard = 1. * TP / (TP + FP + FN)
    return predict, [Global, Accuracy, Class_Avg, Jaccard]
                    
def draw_superpixel(predict, picture, file_name):
    # ---------- super-pixel wise ----------
    for i,p in enumerate(picture):
        w,h = p.getSize()
        image = Image.new('P', (w,h))
        image.putpalette(p.getPalette())
        for j in xrange(w):
            for k in xrange(h):
                index = p.getSIndex(j,k)
                image.putpixel((j,k), predict[i,index])
        name = file_name + str(i) + ".png"
        image.save(name)

def do_forest(boxSize, dataSize, unShuffle, sampleFreq,
              isELMF, isELMAEF, isSTF,
              dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
              numHidden,
              n_superpixels, compactness,
              fileName):
    
    # ----- initialize -----
    print_time('eTRIMS: init', fileName)

    radius = (boxSize - 1) / 2
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize,
                                      shuffle=not unShuffle, name=fileName,
                                      n_superpixels=n_superpixels, compactness=compactness)

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
                           numELM = numHidden, weight=weight, fileName=fileName)
            tree.fit(train_pic=train_pic, test_pic=test_pic, freq=sampleFreq, limit=depthLimit, num=i)

        """
            tree.train(train_pic=train_pic, freq=sampleFreq, limit=depthLimit)
            forest.append(tree)

        print_time('test', fileName)
        forest_test(forest, test_pic, fileName, 'elm_')
        """

    # ----- finish -----
    print_time('eTRIMS: finish', fileName)

    
