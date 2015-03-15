# -*- coding: utf-8 -*-
import os
import Image
import random
import argparse
import datetime
import collections
import numpy as np

##########################################################
## Decision Forest
##########################################################

class DecisionForest(object):
    def __init__(self, num_tree=8, radius=5, num_function=10, condition='gini', seed=123, file_name=None):
        self.num_tree = num_tree
        self.radius = radius
        self.num_function = num_function
        self.condition = condition
        self.np_rng = np.random.RandomState(seed)
        self.node_list = []

    def fit(self, picture, test_picture, d_limit=None, overlap=True):
        # -*- tree_dic -*-
        tree_dic = {}
        index_str = "index"
        dindex_str = "data_index"
        plabel_str = "pre_label"
        pindex_str = "pre_index"
        self.picture = picture
        self.test_picture = test_picture
        
        for t in xrange(self.num_tree):
            # -*- data -*- #
            data = {}
            rate = 0.7
            for i,p in enumerate(picture):
                w,h = p.getSize()
                # <bootstrap>
                sample = random.sample(range(w*h), int(w*h*rate))
                for j in xrange(w):
                    for k in xrange(h):
                        if (j*h + k) in sample:
                            data[i,j,k] = 0
                # </bootstrap>

            # -*- signal -*-
            predict = {}
            for i,p in enumerate(test_picture):
                w,h = p.getSize()
                for j in xrange(w):
                    for k in xrange(h):
                        predict[i,j,k] = 0

            # -*- init tree_dic -*-
            tree_dic[t, index_str] = [0, 1]
            tree_dic[t, dindex_str] = data
            tree_dic[t, plabel_str] = predict
            tree_dic[t, pindex_str] = predict

        # -*- init predict list -*-   
        predict = [(i,j,k) for i,p in enumerate(test_picture) for j in xrange(p.getSize()[0]) for k in xrange(p.getSize()[1])]
        length = len(predict)
            
        current_depth = 0
        isFinish = []
        while (not (all(isFinish) and any(isFinish))) and (d_limit != current_depth):
            current_depth += 1
            isFinish = []

            # -*- update tree_dic -*-
            for t in xrange(self.num_tree):
                s_index, e_index = tree_dic[t, index_str]
                tail_index = e_index
                for index in xrange(s_index, e_index):
                    train = tree_dic[t, dindex_str]
                    test  = tree_dic[t, pindex_str]
                    train_data = [list(x) for x in train.iterkeys() if train[x] == index]
                    test_data  = [list(x) for x in test.iterkeys() if test[x] == index]
                    test_label = tree_dic[t, plabel_str]
                    threshold, d_data, d_label, terminal = self.getOpt(train_data)
                    if not terminal:
                        l_data, r_data = d_data
                        for l in l_data:
                            train[tuple(l)] = tail_index
                        for r in r_data:
                            train[tuple(r)] = tail_index + 1
                        d_test = self.divide(threshold, test_data, self.test_picture)
                        l_label, r_label = d_label
                        l_test, r_test = d_test
                        for l in l_test:
                            test[tuple(l)] = tail_index
                            test_label[tuple(l)] = l_label                          
                        for r in r_test:
                            test[tuple(r)] = tail_index + 1
                            test_label[tuple(r)] = r_label
                        tail_index += 2

                isFinish.append(tail_index == e_index)
                tree_dic[t, index_str] = [e_index, tail_index]

            # -*- get score -*-
            count = 0
            for p in predict:
                label_list = [tree_dic[t, plabel_str][p] for t in xrange(self.num_tree)]
                label = collections.Counter(label_list).most_common()[0][0]
                i,x,y = p
                if test_picture[i].getSignal(x, y) == label:
                    count += 1
            score = count * 1.0 / length
            print_time("depth:%d score = %f" % (current_depth, score), self.file_name)

        node_list = []
        for t in xrange(self.num_tree):
            s_index, e_index = tree_dic[t, index_str]
            node_list.append(e_index)
        self.node_list = node_list

    def getOpt(self, data):
        # -*- unique check -*-
        label_set = set()
        for element in data:
            i,x,y = element
            label_set.put(self.picture[i].getSignal(x,y))
        if len(label_set) == 1:
            return None, None, None, True

        # -*- get optimized threshold from data and self.picture -*-
        limit, count = 50, 0
        while True:
            minimum = None
            threshold = None
            data = None
            label = None
            for n in xrange(self.num_function):
                th = self.generate_threshold(data)
                d_data = self.divide(th, data, self.picture)
                l_data, r_data = d_data
                l_label, r_label = [], []
                for l in l_data:
                    i,x,y = l
                    l_label.append(self.picture[i].getSignal(x,y))
                for r in r_data:
                    i,x,y = r
                    r_label.append(self.picture[i].getSignal(x,y))
                if len(set(l_label)) == 0 or len(set(r_label)) == 0:
                    continue
                gini = self.gini(l_label, r_label)
                if minimum is None or gini < minimum:
                    minimum = gini
                    threshold = th
                    data = d_data
                    label = [l_label, r_label]

            if minimum is None:
                count += 1
                if limit > count:
                    continue
                else:
                    return None, None, None, True

            return threshold, data, label, False
    
    def generate_threshold(self, data):
        selected_dx = self.np_rng.randint(-1*self.radius, 1*self.radius+1)
        selected_dy = self.np_rng.randint(-1*self.radius, 1*self.radius+1)
        selected_c  = self.np_rng.randint(3)

        min_row = 0
        max_row = 255

        theta = self.np_rng.rand() * (max_row - min_row) + min_row
        selected_dim = [selected_dx, selected_dy, selected_c]
        threshold = [selected_dim, theta]
        return threshold
    
    def divide(self, threshold, data, picture):
        lr_data = [[], []]
        selected_dim, theta = threshold
        for element in data:
            i, x, y = element
            dx,dy,c = selected_dim
            val = picture[i].getData(x+dx, y+dy)[c] - theta
            index = val > 0
            lr_data[index].append(element)
        return lr_data

    def gini(self, l_label, r_label):
        set_size = len(l_label) + len(r_label)
        g = 0
        for label in [l_label, r_label]:
            sub_size = len(label)
            counter = collections.Counter(label).most_common()
            for c in counter:
                p = 1. * c[1] / sub_size
                sub = (1. * sub_size / set_size)
                g += sub * p * (1. - p)
        return g
        
    def info(self):
        if self.node_list is None:
            print_time("Information: self.node_length is not defined", self.file_name)
        else:
            print_time("Information: number of node = %s" % str(self.node_list), self.file_name)
        
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
    isF, isEF, isREF, isBEF = t_args
    
    # ----- Decision Tree -----
    if isF:        
        print_time('DecisionTreeForest: init', file_name)
        f = DecisionForest(radius=radius, num_function=num, file_name=file_name)
        
        print_time('DecisionTreeForest: train', file_name)
        f.fit(train_set, test_set, d_limit=d_limit, overlap=cram)
        
        print_time('DecisionTreeForest: info', file_name)
        f.info()
    

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
    etrims_tree(radius=args.radius, size=args.size, d_limit=args.limit, unshuffle=args.unshuffle, cram=args.cram,
                four=args.four, num=args.num, parameter=args.parameter, t_args=t_args, file_name=args.name)

    
