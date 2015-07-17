# -*- coding: utf-8 -*-
import os
#import h5py
import random
import datetime
import operator
import slic as sc
import numpy as np
import collections
from PIL import Image
from extreme import BinaryELMClassifier


def argList(data, index):
    return [list(x) for x in data.iterkeys() if data[x] == index]

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(data, t=1.0):
    # softmax element-wise
    e = np.exp(np.array(data) / t)
    dist = e / np.sum(e)
    return dist

def get_prob(input_dic, t=1.0):
    for key in input_dic.iterkeys():
        input_dic[tuple(key)] = (softmax(input_dic[tuple(key)], t=t)).tolist()
    return input_dic

def neg_grad(input_dic, picture, lr_rate):
    ident = np.identity(8)
    for key in input_dic.iterkeys():
        i,x,y = key
        signal = ident[picture[i].getSignal(x,y) - 1]
        array = signal - lr_rate * np.array(input_dic[tuple(key)])
        input_dic[tuple(key)] = array.tolist()
    return input_dic

def error_info(grad_dic):
    error_list = []
    for key in grad_dic.iterkeys():
        error_list.append(grad_dic[tuple(key)])
    return max(error_list), min(error_list), np.mean(error_list), np.var(error_list)
    
"""
def softmax(data, e_wise=True):
    # softmax element-wise
    e = np.exp(data)
    dist = (e.T / np.sum(e, axis=1)).T if e_wise else e / np.sum(e)
    return dist
"""

##########################################################
##  Criterion
##########################################################

def RSS(data):
    # Criterion for regression tree
    return np.var(data) * len(data)

##########################################################
##  Processing
##########################################################

def predict_draw(est, out_predict, picture, file_name):
    # predict and draw (CURRENTLY TEST DATA ONLY)            
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_pixel(list_dic2dic_dic(out_predict), picture, file_name)
    print_time('GBDT%d_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    print_time('draw_pixel', file_name)    
    draw_pixel(predict, picture, file_name + 'tree%d_pixel' % est)
    # super-pixel
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_superpixel(list_dic2dic_dic(out_predict), picture, file_name)
    print_time('GBDT%d_super: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    print_time('draw_super', file_name)    
    draw_superpixel(predict, picture, file_name + 'tree%d_super' % est)
    
def list_dic2dic_dic(list_dic):
    dic_dic = {}
    for key in list_dic.iterkeys():
        dic_dic[tuple(key)] = {i+1:dic_dic[tuple(key)][i] for i in xrange(0,8)}
    return dic_dic

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

##########################################################
##  Regression Tree
##########################################################

class RegressionTree(object):
    def __init__(self, file_name, train_pic, test_pic, max_depth, max_features, min_leaf_nodes, tree_args=None):
        self.np_rng = np.random.RandomState(123)
        self.file_name = file_name
        self.train_pic = train_pic
        self.test_pic = test_pic
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_nodes = min_leaf_nodes
        
        # add tree_args process
        if not tree_args is None:
            self.func = ['add', 'sub', 'abs', 'uni']
            self.radius = tree_args['radius']
        
    def fit_predict(self, X, y):
        # fit regression tree to X and y
        # predict from test_pic
        # return Output of X and Predict
        X2 = {}
        for i in xrange(len(self.test_pic)):
            w,h = self.test_pic[i].getSize()
            for j in xrange(w):
                for k in xrange(h):
                    X2[i,j,k] = 0
                    
        out_X = {}
        out_X2 = {}
        s_index = 0
        e_index = 1
        node_length = 1
        current_depth = 0
        while s_index < e_index and (current_depth < self.max_depth or self.max_depth is None):
            print_time("depth:%d" % (current_depth), self.file_name)
            current_depth += 1
            print_time("num of node:%d" % (e_index - s_index), self.file_name)
            for index in xrange(s_index, e_index):
                forceTerminal = not current_depth < self.max_depth
                X_data = argList(X, index)
                y_data = [y[tuple(x)] for x in X_data]
                X2_data = argList(X2, index)

                isTerminal, param = self.getOptParam(X_data, y_data, forceTerminal)
                if isTerminal:
                    # get histgram with y_data
                    hist = np.mean(y_data, axis=0)
                    for d in X_data:
                        out_X[tuple(d)] = hist
                    for d in X2_data:
                        out_X2[tuple(d)] = hist
                    # out_X* は引数として用意して
                        # hist を足し合わせるのもいいかも
                        
                if not isTerminal:
                    # X split
                    l_index, r_index = node_length, node_length + 1
                    l_data, r_data = self.split(X_data, param, self.train_pic)
                    for l in l_data:
                        X[tuple(l)] = l_index
                    for r in r_data:
                        X[tuple(r)] = r_index

                    # X2 split
                    l_data, r_data = self.split(X2_data, param, self.test_pic)
                    for l in l_data:
                        X2[tuple(l)] = l_index
                    for r in r_data:
                        X2[tuple(r)] = r_index

                    # add node_length
                    node_length += 2
                    
                    #if current_depth % check_depth == 0 or (s_index < e_index and (current_depth < self.max_depth and not self.max_depth is None)):
                    # checkpoint で何かしたいときの処理                        
                            
                #if current_depth % check_depth == 0 or (s_index < e_index and (current_depth < self.max_depth and not self.max_depth is None)):
                    # checkpoint で何かしたいときの処理
                    
            # update index
            s_index = e_index
            e_index = node_length

        self.node_length = node_length
        print_time("node length:%d" % self.node_length, self.file_name)
        return out_X, out_X2
    
    def fit(self, X, check_depth=10):
        # constructing
        pass

    def predict(self, X, isTest=True):
        # constructing
        pass
    
    def split(self, data, param, data_pic):
        lr_data = [[], []]
        lr_list = map(lambda element:self.split_function(element, param, data_pic) > 0, data)
        for i, lr in enumerate(lr_list):
            lr_data[lr].append(data[i])

        l_data, r_data = lr_data
        return l_data, r_data
    
    def split_Xy(self, X_data, y_data, param, data_pic):
        l_data, r_data = self.split(X_data, param, data_pic)
        l_label = [y_data[X_data.index(d)] for d in l_data]
        r_label = [y_data[X_data.index(d)] for d in r_data]
        return l_data, l_label, r_data, r_label
    
    def getOptParam(self, X_data, y_data, forceTerminal):
        # check isTerminal
        if len(X_data) <= self.min_leaf_nodes or forceTerminal:
            # terminal
            return True, None
        
        # find optimized parameter
        opt_rss = None
        opt_param = None
        for i in xrange(self.max_features):
            #print_time('th: %i' % i, self.fileName)
            param = self.generate_threshold(X_data)
            l_X, l_y, r_X, r_y = self.split_Xy(X_data, y_data, param, self.train_pic)
            rss = RSS(l_y) + RSS(r_y)
            if len(l_X) == 0 or len(r_X) == 0:
                continue
            if opt_rss is None or rss < opt_rss:
                opt_param = param
                opt_rss = rss
        if opt_param is None:
            # terminal
            return True, None
        # inner
        return False, opt_param

    # ---------- Eigen method ----------- 
    def split_function(self, element, param, picture):
        i, x, y = element
        f, pos, theta = param
        [x1, y1, c1], [x2, y2, c2] = pos

        if f == 'add':
            return picture[i].getData(x + x1, y + y1)[c1] + picture[i].getData(x + x2, y + y2)[c2] - theta
        if f == 'sub':
            return picture[i].getData(x + x1, y + y1)[c1] - picture[i].getData(x + x2, y + y2)[c2] - theta
        if f == 'abs':
            return abs(picture[i].getData(x + x1, y + y1)[c1] - picture[i].getData(x + x2, y + y2)[c2]) - theta
        if f == 'uni':
            return picture[i].getData(x + x1, y + y1)[c1] - theta

    # ---------- Eigen method -----------
    def generate_threshold(self, data):
        f = self.func[random.randint(0, len(self.func)-1)]
        theta = random.random()
        x1, y1, x2, y2 = [random.randint(-1 * self.radius, self.radius) for col in xrange(4)]
        c1, c2 = [random.randint(0, 2) for col in xrange(2)]
        pos = [[x1, y1, c1], [x2, y2, c2]]
        theta = random.random()
        if f == 'add':
            theta = random.random() * 2
        if f == 'sub':
            theta = random.random() * 2 - 1
        if f == 'abs':
            theta = random.random()
        if f == 'uni':
            theta = random.random()        
        return f, pos, theta        

##########################################################
##  ELM Regression Tree
##########################################################

class ELMRegressionTree(RegressionTree):
    def __init__(self, file_name, train_pic, test_pic, max_depth, max_features, min_leaf_nodes, tree_args=None):
        super(ELMRegressionTree, self).__init__(file_name, train_pic, test_pic, max_depth,
                                                max_features, min_leaf_nodes, None)
        # add tree_args process
        if not tree_args is None:
            self.radius = tree_args['radius']
            self.elm_hidden = tree_args['elm_hidden']
                 
    # ---------- Eigen method -----------
    def split_function(self, element, param, picture):
        i,x,y = element
        weight, bias, beta = param
        crop = picture[i].cropData(x, y, self.radius)
        hidden = sigmoid(np.dot(weight.T, crop) + bias)
        output = np.dot(beta.T, hidden) # sigmoid(np.dot(hidden, beta))
        return output - 0.5 # constant theta
        
    # ---------- Eigen method -----------
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
        
##########################################################
##  Gradient Boosting Classifier
##########################################################

class GradientBoostingClassifier(object):
    def __init__(self, file_name, learning_rate, n_estimators,
                 max_depth, sample, freq, max_features, min_leaf_nodes,
                 alpha=0.9, verpose=0, 
                 tree_type='reg', tree_args={}):
        # loss
        # min_sample_split
        # min_sample_leaf
        # sub_sample
        # verpose
        # max_leaf_nodes
        # warm_start

        self.file_name = file_name
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample = sample
        self.freq = freq
        self.max_features = max_features
        self.min_leaf_nodes = min_leaf_nodes
        self.alpha = alpha
        self.tree_type = tree_type
        self.tree_args = tree_args
        
    def fit_predict(self, train_pic, test_pic):
        # label-weight 問題をどう処理するか(線形の逆数だとうまくいかないと思う)
        # hist 処理はここでする、yのリスト構造を1~8のラベルの辞書に変換しておく
        #  {i:list_hist[i-1] for i in xrange(1,9)}
        # sampling と sub-sampling をわけて実装(stochastic)
        
        self.train_pic = train_pic
        self.test_pic = test_pic
        sample = {}
                               
        for i in xrange(len(self.train_pic)):
            w,h = self.train_pic[i].getSize()
            for j in xrange(0, w, self.freq):
                for k in xrange(0, h, self.freq):
                    # bootstrap
                    if random.random() < self.sample and not self.train_pic[i].getSignal(j,k):
                        sample[i,j,k] = 0

        # start with initial model
        label_list = []
        for key in sample.iterkeys():
            i,x,y = key
            label_list.append(self.train_pic[i].getSignal(x,y))
        poll_dic = collections.Counter(label_list)
        poll = [poll_dic[col] for col in xrange(1,9)]
        initial_output = np.array(poll) / np.sum(poll)
        # initial_output は一様分布にしてみたらどうか

        out_sample = {}
        for key in out_sample.iterkeys():
            out_sample[tuple(key)] = initial_output # initial output

        out_test = {}
        for i in xrange(len(self.test_pic)):
            w,h = self.test_pic[i].getSize()
            for j in xrange(0, w, self.sample):
                for k in xrange(0, h, self.sample):
                    out_test[i,j,k] = initial_output # initial output
                
        # iterate self.n_estimators times
        for est in xrange(self.n_estimators):
            # calcurate probability
            out_sample = get_prob(out_sample)
            out_test = get_prob(out_test)

            # predict and draw (CURRENTLY TEST DATA ONLY)
            predict_draw(est, out_test)

            # calcurate negative gradient
            grad_sample = neg_grad(out_sample, self.train_pic, self.learning_rate)

            # calcurate learning error (optional)
            max_error, min_error, mean_error, var_error = error_info(grad_sample)
            print_time('GBDT%d_error_info: Max %f Min %f Mean %f var %f' % (est, max_error, min_error, mean_error, var_error), self.file_name)

            # fit a regression tree to negative gradient
            tree_obj = REG_TREES[self.tree_type]
            tree = tree_obj(self.file_name, self.train_pic, self.test_pic, self.max_depth, self.max_features, self.min_leaf_nodes, self.tree_args)
            out_X, out_X2 = tree.fit_predict(sample, grad_sample)
            
            # update output (out_sample + out_X and out_test + out_X2)
            for key in out_sample.iterkeys():
                out_sample[tuple(key)] = [out_sample[tuple(key)][i] + out_X[tuple(key)][i] for i in xrange(0, 8)]
            for key in out_test.iterkeys():
                out_test[tuple(key)] = [out_test[tuple(key)][i] + out_X2[tuple(key)][i] for i in xrange(0, 8)]

        # calcurate probability
        out_sample = get_prob(out_sample)
        out_test = get_prob(out_test)
            
        # predict and draw (CURRENTLY TEST DATA ONLY)            
        predict_draw(self.n_estimators, out_test)

        # calcurate negative gradient
        grad_sample = neg_grad(out_sample, self.train_pic, self.learning_rate)
        
        # calcurate learning error (optional)
        max_error, min_error, mean_error, var_error = error_info(grad_sample)
        print_time('GBDT%d_error_info: Max %f Min %f Mean %f var %f' % (self.n_estimators, max_error, min_error, mean_error, var_error), self.file_name)

        # done
        print_time('done.', self.file_name)       

    
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
##  Main Function
##########################################################

REG_TREES = {'reg': RegressionTree, 'elm': ELMRegressionTree} 
IS_TRAIN_DATA = False

def do_forest(boxSize, dataSize, unShuffle, sampleFreq,
              isREG, isELMREG,
              dataPerTree, depthLimit, numThreshold, numTree, sampleSize,
              numHidden,
              n_superpixels, compactness,
              file_name):
    
    # ----- initialize -----
    print_time('eTRIMS: init', file_name)

    radius = (boxSize - 1) / 2
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize,
                                      shuffle=not unShuffle, name=file_name,
                                      n_superpixels=n_superpixels, compactness=compactness)

    print_parameter([boxSize, dataSize, unShuffle, sampleFreq], file_name)
    print_parameter([isREG, isELMREG], file_name)
    print_parameter([dataPerTree, depthLimit, numThreshold, numTree], file_name)
    print_parameter([numHidden], file_name)
    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d'
               % (radius, str(depthLimit), dataSize, numThreshold), file_name)

    # compute label weight
    #weight = compute_weight(train_pic)

    if isREG:
        print_time('GBDT Regression', file_name)
        learning_rate = 0.9
        n_estimators = 10
        max_depth = 10
        sample = 0.25
        freq = 5
        max_features = 10
        min_leaf_nodes = None
        alpha = 0.8
        verpose = None
        tree_type = 'reg'
        tree_args = {'radius':15}
        GradientBoostingClassifier(file_name, learning_rate, n_estimators, max_depth, sample, freq,
                                   max_features, min_leaf_nodes, alpha, verpose, tree_type, tree_args)

    # ----- finish -----
    print_time('eTRIMS: finish', file_name)
