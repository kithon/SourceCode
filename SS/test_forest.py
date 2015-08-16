# -*- coding: utf-8 -*-
import os
import h5py
import random
import datetime
import operator
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

def neg_grad(input_dic, h5file):
    type = 'train'
    ident = np.identity(8)
    neg_dic = {}
    for key in input_dic.iterkeys():
        i,x,y = key
        signal = ident[h5_signal(h5file,type,i,x,y) - 1]
        array = signal - np.array(input_dic[tuple(key)])
        neg_dic[tuple(key)] = array.tolist()
    return neg_dic

def error_info(grad_dic):
    error_list = []
    for key in grad_dic.iterkeys():
        error_list.append(grad_dic[tuple(key)])
    return np.max(error_list), np.min(error_list), np.mean(error_list), np.var(error_list)
    
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

def predict_draw(est, out_predict, h5file, tree_type, file_name):
    # predict and draw (CURRENTLY TEST DATA ONLY)            
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_pixel(list_dic2dic_dic(out_predict), h5file, 'test', file_name)
    print_time('GBDT_%s_%d_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (tree_type, est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    print_time('draw_pixel', file_name)    
    draw_pixel(predict, h5file, 'test', 'tree_%s_%d_pixel' % (tree_type, est))
    # super-pixel
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_superpixel(list_dic2dic_dic(out_predict), h5file, 'test', file_name)
    print_time('GBDT_%s_%d_super: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (tree_type, est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    print_time('draw_super', file_name)    
    draw_superpixel(predict, h5file, 'test', 'tree_%s_%d_super' % (tree_type, est))
    
def list_dic2dic_dic(list_dic):
    dic_dic = {}
    for key in list_dic.iterkeys():
        dic_dic[tuple(key)] = {i+1:list_dic[tuple(key)][i] for i in xrange(0,8)}
    return dic_dic

def predict_pixel(hist, h5file, type, fileName):
    # ---------- pixel wise ----------q
    TP, TN, FP, FN = 0, 0, 0, 0
    one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0
    predict = {}
    for i in xrange(h5file[type + '/num'].value):
        width, height = h5_size(h5file, type, i)
        for j in xrange(width):
            for k in xrange(height):
                label = h5_signal(h5file,type,i,j,k)
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

def draw_pixel(predict, h5file, type, file_name):
    # ---------- pixel wise ----------
    for i in xrange(h5file[type + '/num'].value):
        w, h = h5_size(h5file, type, i)
        image = Image.new('P', (w,h))
        image.putpalette(h5file['option/palette'].value)
        for j in xrange(w):
            for k in xrange(h):
                image.putpixel((j,k), predict[i,j,k])
        name = file_name + str(i) + ".png"
        image.save(name)
    
def predict_superpixel(hist, h5file, type, fileName):
    # ---------- super-pixel wise ----------
    # compress hist
    length = 0
    super_hist = {}
    for i in xrange(h5file[type + '/num'].value):
        width, height = h5_size(h5file, type, i)
        length += (width * height)
        for index in xrange(h5_ssize(h5file,type,i)):
            super_hist[i,index] = {col:0 for col in xrange(1,9)}         
        for j in xrange(width):
            for k in xrange(height):
                index = h5_sindex(h5file,type,i,j,k)
                for col in xrange(1,9):
                    super_hist[i,index][col] = super_hist[i,index][col] + hist[i,j,k][col] 

    # predict
    predict = {}
    for i in xrange(h5file[type + '/num'].value):
        for index in xrange(h5_ssize(h5file,type,i)):
            predict[i,index] = max(super_hist[i,index].iteritems(), key=operator.itemgetter(1))[0]

    # count
    TP, TN, FP, FN = 0, 0, 0, 0
    one_TP, one_TN, one_FP, one_FN = 0, 0, 0, 0
    for i in xrange(h5file[type + '/num'].value):
        width, height = h5_size(h5file, type, i)
        for j in xrange(width):
            for k in xrange(height):
                label = h5_signal(h5file,type,i,j,k)
                index = h5_sindex(h5file,type,i,j,k)
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
                    
def draw_superpixel(predict, h5file, type, file_name):
    # ---------- super-pixel wise ----------
    for i in xrange(h5file[type + '/num'].value):
        w, h = h5_size(h5file, type, i)
        image = Image.new('P', (w,h))
        image.putpalette(h5file['option/palette'].value)
        for j in xrange(w):
            for k in xrange(h):
                index = h5_sindex(h5file, type, i, j, k)
                image.putpixel((j,k), predict[i,index])
        name = file_name + str(i) + ".png"
        image.save(name)

##########################################################
##  Regression Tree
##########################################################

class RegressionTree(object):
    def __init__(self, file_name, max_depth, max_features, min_leaf_nodes, tree_args=None):
        self.np_rng = np.random.RandomState(123)
        self.file_name = file_name
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_nodes = min_leaf_nodes
        
        # add tree_args process
        if not tree_args is None:
            self.func = ['add', 'sub', 'abs', 'uni']
            self.radius = tree_args['radius']
        
    def fit_predict(self, X, y, h5file, weight=None):
        # fit regression tree to X and y
        # predict from test_pic
        # return Output of X and Predict
        X2 = {}
        for i in xrange(h5file['test/num'].value):
            w,h = h5_size(h5file,'test',i)
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

                isTerminal, param = self.getOptParam(X_data, y_data, forceTerminal, h5file, 'train')
                if isTerminal:
                    # get histgram with y_data
                    hist = np.mean(y_data, axis=0)
                    if not weight is None:
                        hist *= np.array(weight)
                    for d in X_data:
                        out_X[tuple(d)] = hist
                    for d in X2_data:
                        out_X2[tuple(d)] = hist
                    # out_X* は引数として用意して
                        # hist を足し合わせるのもいいかも
                        
                if not isTerminal:
                    # X split
                    l_index, r_index = node_length, node_length + 1
                    l_data, r_data = self.split(X_data, param, h5file, 'train')
                    for l in l_data:
                        X[tuple(l)] = l_index
                    for r in r_data:
                        X[tuple(r)] = r_index

                    # X2 split
                    l_data, r_data = self.split(X2_data, param, h5file, 'test')
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
    
    def split(self, data, param, h5file, type):
        lr_data = [[], []]
        lr_list = map(lambda element:self.split_function(element, param, h5file, type) > 0, data)
        for i, lr in enumerate(lr_list):
            lr_data[lr].append(data[i])

        l_data, r_data = lr_data
        return l_data, r_data
    
    def split_Xy(self, X_data, y_data, param, h5file, type):
        l_data, r_data = self.split(X_data, param, h5file, type)
        l_label = [y_data[X_data.index(d)] for d in l_data]
        r_label = [y_data[X_data.index(d)] for d in r_data]
        return l_data, l_label, r_data, r_label
    
    def getOptParam(self, X_data, y_data, forceTerminal, h5file, type):
        # check isTerminal
        if len(X_data) <= self.min_leaf_nodes or forceTerminal:
            # terminal
            print_time("forcely1", self.file_name)
            return True, None
        
        # find optimized parameter
        opt_rss = None
        opt_param = None
        for i in xrange(self.max_features):
            #print_time('th: %i' % i, self.fileName)
            param = self.generate_threshold(X_data, y_data, h5file, type)
            l_X, l_y, r_X, r_y = self.split_Xy(X_data, y_data, param, h5file, type)
            rss = RSS(l_y) + RSS(r_y)
            if len(l_X) == 0 or len(r_X) == 0:
                continue
            if opt_rss is None or rss < opt_rss:
                opt_param = param
                opt_rss = rss
        if opt_param is None:
            # terminal
            print_time("forcely2", self.file_name)
            return True, None
        # inner
        return False, opt_param

    # ---------- Eigen method ----------- 
    def split_function(self, element, param, h5file, type):
        i, x, y = element
        f, pos, theta = param
        [x1, y1, c1], [x2, y2, c2] = pos

        if f == 'add':
            return h5_data(h5file,type,i,x + x1, y + y1)[c1] + h5_data(h5file,type,i,x + x2, y + y2)[c2] - theta
        if f == 'sub':
            return h5_data(h5file,type,i,x + x1, y + y1)[c1] - h5_data(h5file,type,i,x + x2, y + y2)[c2] - theta
        if f == 'abs':
            return abs(h5_data(h5file,type,i,x + x1, y + y1)[c1] - h5_data(h5file,type,i,x + x2, y + y2)[c2] - theta) - theta
        if f == 'uni':
            return  h5_data(h5file,type,i,x + x1, y + y1)[c1] - theta

    # ---------- Eigen method -----------
    def generate_threshold(self, data, signal, h5file, type):
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
    def __init__(self, file_name, max_depth, max_features, min_leaf_nodes, tree_args=None):
        super(ELMRegressionTree, self).__init__(file_name, max_depth, max_features, min_leaf_nodes, None)
        # add tree_args process
        if not tree_args is None:
            self.radius = tree_args['radius']
            self.sample_size = tree_args['sample_size']
            self.elm_hidden = tree_args['elm_hidden']

    # original method
    def split(self, data, param, h5file, type):
        lr_data = [[], []]
        lr_list = self.split_function_batch(data, param, h5file, type)
        #lr_list = map(lambda element:self.split_function(element, param, h5file, type) > 0, data)
        for i, lr in enumerate(lr_list):
            lr_data[lr].append(data[i])

        l_data, r_data = lr_data
        return l_data, r_data
    
    def split_function_batch(self, data, param, h5file, type):
        lr_list = []
        batch_size = 50000
        weight, bias, beta = param
        for batch_data in [data[col:col+batch_size] for col in range(0, len(data), batch_size)]:
            batch_input = [h5_crop(h5file,type,i,x,y) for (i,x,y) in batch_data]
            hidden = sigmoid(np.dot(batch_input, weight) + bias)
            output = np.dot(hidden, beta) - 0.5 # sigmoid(np.dot(hidden, beta))
            lr_list += map(lambda input: input> 0, output)
        return lr_list
       
    def split_function(self, element, param, h5file, type):
        i,x,y = element
        weight, bias, beta = param
        crop = h5_crop(h5file,type,i,x,y)
        hidden = sigmoid(np.dot(weight.T, crop) + bias)
        output = np.dot(beta.T, hidden) # sigmoid(np.dot(hidden, beta))
        return output - 0.5 # constant theta
        
    def generate_threshold(self, data, signal, h5file, type):
        # crop data
        sample_input, label = [], []
        num = min(len(data), self.sample_size)
        sample_index = random.sample(data, num)
        for temp in sample_index:
            i,x,y = temp
            sample_input.append(h5_crop(h5file,type,i,x,y))
            label.append(h5_signal(h5file,type,i,x,y))
            #print_time(signal[data.index(temp)], self.file_name)
            #print_time(h5_signal(h5file,type,i,x,y), self.file_name)

        # label
        label_index = []
        numL, numR = 0, 0
        for l in collections.Counter(label).most_common():
            if numL < numR:
                numL += l[1]
                label_index.append(l[0])
                #print_time("right", self.file_name)
            else:
                numR += l[1]
                #print_time("left", self.file_name)

        sample_signal = [1 if l in label_index else 0 for l in label]
            
        # debug
        print_time(collections.Counter(label), self.file_name)

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

        
    def fit_predict(self, h5file, boxSize, palette, dataSize, train_size, test_size):#train_pic, test_pic):
        # label-weight 問題をどう処理するか(線形の逆数だとうまくいかないと思う)
        # hist 処理はここでする、yのリスト構造を1~8のラベルの辞書に変換しておく
        #  {i:list_hist[i-1] for i in xrange(1,9)}
        # sampling と sub-sampling をわけて実装(stochastic)
        
        sample = {}        
        for i in xrange(train_size):
            w,h = h5_size(h5file, 'train', i)
            for j in xrange(0, w, self.freq):
                for k in xrange(0, h, self.freq):
                    # bootstrap
                    if random.random() < self.sample and h5_signal(h5file, 'train', i, j, k):
                        sample[i,j,k] = 0

        weight = compute_weight(h5file, 'train', sample)

        # start with initial model
        label_list = []
        for key in sample.iterkeys():
            i,x,y = key
            label_list.append(h5_signal(h5file,'train',i,x,y))
        poll_dic = collections.Counter(label_list)
        poll = [poll_dic[col] for col in xrange(1,9)]
        initial_output = np.array(poll) / np.sum(poll)
        # initial_output は一様分布にしてみたらどうか

        out_sample = {}
        for key in sample.iterkeys():
            out_sample[tuple(key)] = initial_output # initial output

        out_test = {}
        for i in xrange(test_size):
            w,h = h5_size(h5file,'test',i)
            for j in xrange(0, w):
                for k in xrange(0, h):
                    out_test[i,j,k] = initial_output # initial output
                
        # iterate self.n_estimators times
        for est in xrange(self.n_estimators):
            # calcurate probability
            print_time('len : %d' % (len(out_sample)), self.file_name)
            out_sample = get_prob(out_sample)
            out_test = get_prob(out_test)

            # predict and draw (CURRENTLY TEST DATA ONLY)
            predict_draw(est, out_test, h5file, self.tree_type, self.file_name)

            # calcurate negative gradient
            grad_sample = neg_grad(out_sample, h5file)

            # calcurate learning error (optional)
            max_error, min_error, mean_error, var_error = error_info(grad_sample)
            print_time('GBDT%d_error_info: Max %f Min %f Mean %f Var %f' % (est, max_error, min_error, mean_error, var_error), self.file_name)

            # initialize sample
            for key in sample.iterkeys():
                sample[tuple(key)] = 0
            
            # fit a regression tree to negative gradient
            tree_obj = REG_TREES[self.tree_type]
            tree = tree_obj(self.file_name, self.max_depth, self.max_features, self.min_leaf_nodes, self.tree_args)
            out_X, out_X2 = tree.fit_predict(sample, grad_sample, h5file, weight)
            
            # update output (out_sample + out_X and out_test + out_X2)
            for key in out_sample.iterkeys():
                out_sample[tuple(key)] = [out_sample[tuple(key)][i] + self.learning_rate * out_X[tuple(key)][i] for i in xrange(0, 8)]
            for key in out_test.iterkeys():
                out_test[tuple(key)] = [out_test[tuple(key)][i] + self.learning_rate * out_X2[tuple(key)][i] for i in xrange(0, 8)]

        # calcurate probability
        out_sample = get_prob(out_sample)
        out_test = get_prob(out_test)
            
        # predict and draw (CURRENTLY TEST DATA ONLY)
        predict_draw(self.n_estimators, out_test, h5file, self.tree_type, self.file_name)

        # calcurate negative gradient
        grad_sample = neg_grad(out_sample, h5file)
        
        # calcurate learning error (optional)
        max_error, min_error, mean_error, var_error = error_info(grad_sample)
        print_time('GBDT%d_error_info: Max %f Min %f Mean %f var %f' % (self.n_estimators, max_error, min_error, mean_error, var_error), self.file_name)

        # done
        print_time('done.', self.file_name)

    
##########################################################
##  h5py
##########################################################
# ----- element type 1 -----
def get_e1(type, i):
    return type + '/' + str(i)
    
def h5_name(h5file, type, i):
    element = get_e1(type, i) + '/name'
    return h5file[element].value

def h5_size(h5file, type, i):
    element = get_e1(type, i) + '/size'
    return h5file[element].value

def h5_ssize(h5file, type, i):
    element = get_e1(type, i) + '/ssize'
    return h5file[element].value

def h5_scenter(h5file, type, i):
    element = get_e1(type, i) + '/scenter'
    return h5file[element].value

# ----- element type 2 -----
def get_e2(type, i, x, y):
    return type + '/' + str(i) + '/' + str(x) + '/' + str(y)

def h5_data(h5file, type, i, x, y):
    element = get_e2(type,i,x,y) + '/data'
    return h5file[element].value

def h5_signal(h5file, type, i, x, y):
    element = get_e2(type,i,x,y) + '/signal'
    return h5file[element].value

def h5_crop(h5file, type, i, x, y):
    element = get_e2(type,i,x,y) + '/crop'
    return h5file[element].value

def h5_sindex(h5file, type, i, x, y):
    element = get_e2(type,i,x,y) + '/sindex'
    return h5file[element].value

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

def compute_weight(h5file, type, data=None):
    # compute label weight from train picture
    label = []
    if data is None:
        for i in xrange(h5file[type + '/num'].value):
            w,h = h5_size(h5file,type,i)
            for x in xrange(w):
                for y in xrange(h):
                    label.append(h5_signal(h5file,type,i,x,y))
    else:
        for key in data.iterkeys():
            i,x,y = key
            label.append(h5_signal(h5file,type,i,x,y))

    label_weight = {}
    for l in collections.Counter(label).most_common():
        label_weight[l[0]] = 1. / l[1]
    return label_weight
        
##########################################################
##  Main Function
##########################################################

REG_TREES = {'reg': RegressionTree, 'elm': ELMRegressionTree} 
IS_TRAIN_DATA = False

def do_forest(isREG, isELMREG,
              n_estimator, max_depth, sample_pertree, sample_freq,
              max_features, min_leaf_nodes, alpha, learning_rate, verpose,
              input_file, result_dir, file_name):
    
    # ----- initialize -----
    print_time('eTRIMS: init', file_name)

    # ----- load h5file -----
    h5file = h5py.File(input_file,"r")
    option_dir = 'option'
    train_dir  = 'train'
    test_dir   = 'test'
    
    radius = h5file[option_dir + '/radius'].value
    boxSize = radius * 2 + 1
    palette = h5file[option_dir + '/palette'].value
    dataSize = h5file[option_dir + '/num'].value
    train_size = h5file[train_dir + '/num'].value
    test_size  = h5file[test_dir  + '/num'].value

    # config REG
    reg_args = {'radius': (boxSize - 1) / 2}

    # config ELMREG
    elmreg_args = {'radius': (boxSize - 1) / 2,
                   'sample_size': boxSize * boxSize,
                   'elm_hidden': boxSize * boxSize * 3 * 2}

    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d'
               % (radius, str(max_depth), dataSize, max_features), file_name)

    # compute label weight
    #weight = compute_weight(train_pic)

    if isREG:
        print_time('GBDT Regression', file_name)
        gbdt = GradientBoostingClassifier(file_name, learning_rate, n_estimator,
                                          max_depth, sample_pertree, sample_freq,
                                          max_features, min_leaf_nodes, alpha, verpose, 'reg', reg_args)
        gbdt.fit_predict(h5file, boxSize, palette, dataSize, train_size, test_size)

    if isELMREG:
        print_time('GBDT ELMRegression', file_name)
        gbdt = GradientBoostingClassifier(file_name, learning_rate, n_estimator,
                                          max_depth, sample_pertree, sample_freq,
                                          max_features, min_leaf_nodes, alpha, verpose, 'elm', elmreg_args)
        gbdt.fit_predict(h5file, boxSize, palette, dataSize, train_size, test_size)

    # ----- finish -----
    h5file.close()
    print_time('eTRIMS: finish', file_name)

    
