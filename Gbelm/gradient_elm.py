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
from extreme import ELMRegressor


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

def get_grad(value, signal):
    grad_list = signal - np.array(value)
    return grad_list.tolist()

def neg_grad(input_dic, picture):
    ident = np.identity(8)
    neg_dic = {}
    for key in input_dic.iterkeys():
        i,x,y = key
        signal = ident[picture[i].getSignal(x,y) - 1]
        array = signal - np.array(input_dic[tuple(key)])
        neg_dic[tuple(key)] = array.tolist()
    return neg_dic

def error_info(grad_dic):
    error_list = []
    for key in grad_dic.iterkeys():
        error_list.append(grad_dic[tuple(key)])
    return np.max(error_list), np.min(error_list), np.mean(error_list), np.var(error_list)
    
##########################################################
##  Processing
##########################################################

def predict_draw(est, out_predict, picture, prefix, file_name):
    # predict and draw
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_pixel(list_dic2dic_dic(out_predict), picture, file_name)
    print_time('ELM_%s_%d_pixel: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (prefix, est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    draw_pixel(predict, picture, 'elm_%s_%d_pixel' % (prefix, est))
    # super-pixel
    predict, [Global, Accuracy, Class_Avg, Jaccard] = predict_superpixel(list_dic2dic_dic(out_predict), picture, file_name)
    print_time('ELM_%s_%d_super: Global %f Accuracy %f Class_Avg %f Jaccard %f' % (prefix, est, Global, Accuracy, Class_Avg, Jaccard), file_name)
    draw_superpixel(predict, picture, 'elm_%s_%d_super' % (prefix, est))
    
def list_dic2dic_dic(list_dic):
    dic_dic = {}
    for key in list_dic.iterkeys():
        dic_dic[tuple(key)] = {i+1:list_dic[tuple(key)][i] for i in xrange(0,8)}
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
##  Pic
##########################################################

class Pic(object):
    __slots__ = ['data', 'signal', 'spixel', 'palette',
                 'slength', 'scenter', 'sdic', 'w', 'h', 'radius']
    def __init__(self, data, signal, spixel, radius):
        self.w, self.h = data.size
        self.palette = signal.getpalette()
        self.radius = radius
        self.setData(data)
        self.setSignal(signal)
        self.setSpixel(spixel)

    def setData(self, data):
        data_list = []
        for j in xrange(-self.radius, self.w + self.radius + 1):
            temp = []
            if j < 0:
                x = -1 * j
            elif j < self.w:
                x = j
            else:
                x = j - self.w
            for k in xrange(-self.radius, self.h + self.radius + 1):
                if k < 0:
                    y = -1 * k
                elif k < self.h:
                    y = k
                else:
                    y = k - self.h
                    
                temp.append(list(data.getpixel((x,y))))
            data_list.append(temp)
        self.data = np.array(data_list)
        
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
        return self.data[x + self.radius][y + self.radius]

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
        return self.data[x + self.radius][y + self.radius]

    def getSSignal(self, index):
        return self.sdic[index]
    
    def cropData(self, x, y, radius):
        crop = self.data[x:x + 2*self.radius + 1, y:y + 2*radius + 1]
        crop = np.resize(crop, (3 * (2*radius+1) ** 2,))
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
        picture = Pic(image, annotation, spixel, radius)
        test_or_train[index].append(picture)

        # print filename
        print_time("eTRIMS: %s" % file_name, name)

    # ----- finish -----
    print_parameter(train_index, name)
    print_time("eTRIMS: train=%d test=%d" % (len(train_set), len(test_set)), name)
    return train_set, test_set

def compute_weight(data_pic, data=None):
    # compute label weight from train picture
    label = []
    if data is None:
        for i,p in enumerate(data_pic):
            w,h = p.getSize()
            for x in xrange(w):
                for y in xrange(h):
                    label.append(data_pic[i].getSignal(x,y))
    else:
        for key in data.iterkeys():
            i,x,y = key
            label.append(data_pic[i].getSignal(x,y))

    label_weight = {}
    for l in collections.Counter(label).most_common():
        label_weight[l[0]] = 1. / l[1]
    return [label_weight[col] for col in xrange(1,9)]
        
##########################################################
##  Main Function
##########################################################

def fit_elm(file_name, train_pic, radius, elm_hidden, learning_rate, n_estimator, sample_perelm, sample_freq):
    # initialize predict
    print_time('fit_elm: initialize predict', file_name)
    label = []
    for i,p in enumerate(train_pic):
        w,h = p.getSize()
        for x in xrange(w):
            for y in xrange(h):
                label.append(train_pic[i].getSignal(x,y))

    label_weight = {}
    for l in collections.Counter(label).most_common():
        label_weight[l[0]] = l[1]
    initial_output = [label_weight[col] for col in xrange(1,9)] # 一様分布でも良い
    #weight = compute_weight(train_pic)

    # initialize out_train
    print_time('fit_elm: initialize out_train', file_name)
    out_train = {}
    for i in xrange(len(train_pic)):
        w,h = train_pic[i].getSize()
        for j in xrange(w):
            for k in xrange(h):
                out_train[i,j,k] = initial_output
    train_prob = get_prob(out_train)

    # fit train_pic
    print_time('fit_elm: fit train_pic', file_name)
    elm_list = []
    ident = np.identity(8)
    for est in xrange(n_estimator):
        #########################################
        # Stochastic Gradient Boosting
        #########################################
        print_time('fit_elm: %dth train' % est, file_name)

        # create elm_input and elm_signal
        print_time('fit_elm: create elm_input and elm_signal', file_name)
        elm_input, elm_signal = [], []
        for i in xrange(len(train_pic)):
            w,h = train_pic[i].getSize()
            for x in xrange(0, w, sample_freq):
                for y in xrange(0, h, sample_freq):
                    label = train_pic[i].getSignal(x,y)
                    if random.random() < sample_perelm and label:
                        signal = ident[label - 1]
                        elm_input.append(train_pic[i].cropData(x,y,radius))
                        elm_signal.append(get_grad(train_prob[i,x,y], signal))
        # elm fit
        print_time('fit_elm: elm fit', file_name)
        print_time('input %d  hidden %d  output 8  data %d' % (3*(2*radius+1)**2, elm_hidden, len(elm_input)), file_name)
        elm = ELMRegressor(n_hidden=elm_hidden)
        #elm = MLELMRegressor(n_hidden=[elm_input, elm_hidden, 8])
        elm.fit(elm_input, elm_signal)

        # elm predict
        print_time('fit_elm: elm predict1', file_name)
        for key in out_train.iterkeys():
            i,x,y = key
            output = elm.predict(train_pic[i].cropData(x,y,radius))
            out_train[tuple(key)] = [out_train[tuple(key)][col] + learning_rate * output[col] for col in xrange(0,8)]
        train_prob = get_prob(out_train) # use softmax

        # draw picture and evaluate accuracy
        print_time('fit_elm: draw picture and evaluate accuracy', file_name)
        predict_draw(est, train_prob, train_pic, 'train', file_name)

        elm_list.append(elm)
    print_time('fit_elm: done.', file_name)
    return initial_output, elm_list


def predict_elm(file_name, test_pic, radius, initial_output, learning_rate, elm_list):
    # initialize out_test
    print_time('predict_elm: initialize out_test', file_name)
    out_test = {}
    for i in xrange(len(test_pic)):
        w,h = test_pic[i].getSize()
        for j in xrange(w):
            for k in xrange(h):
                out_test[i,j,k] = initial_output
    test_prob = get_prob(out_test)

    # predict test_pic
    print_time('predict_elm: predict test_pic', file_name)
    for est, elm in enumerate(elm_list):
        print_time('predict_elm: %dth train' % est, file_name)
        
        # elm predict
        print_time('predict_elm: elm predict', file_name)
        for key in out_test.iterkeys():
            i,x,y = key
            output = elm.predict(test_pic[i].cropData(x,y,radius))
            out_test[tuple(key)] = [out_test[tuple(key)][col] + learning_rate * output[col] for col in xrange(0,8)]
        test_prob = get_prob(out_test)

        # draw picture and evaluate accuracy
        print_time('predict_elm: draw_picture and evaluate accuracy', file_name)
        predict_draw(est, test_prob, test_pic, 'test', file_name)

    print_time('predict_elm: done.', file_name)

def do_elm(boxSize, dataSize, unShuffle,
           n_estimator, sample_perelm, sample_freq,
           elm_hidden, learning_rate, 
           n_superpixels, compactness,
           file_name):
    
    # ----- initialize -----
    print_time('eTRIMS: init', file_name)

    radius = (boxSize - 1) / 2
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize,
                                      shuffle=not unShuffle, name=file_name,
                                      n_superpixels=n_superpixels, compactness=compactness)

    initial_output, elm_list = fit_elm(file_name, train_pic, radius, elm_hidden, learning_rate, n_estimator, sample_perelm, sample_freq)
    predict_elm(file_name, test_pic, radius, initial_output, learning_rate, elm_list)

    # ----- finish -----
    print_time('eTRIMS: finish', file_name)


