# -*- coding: utf-8 -*-
import os
import sys
import scipy.io
import datetime
import numpy as np
from PIL import Image
from tree import DecisionTree, ExtremeDecisionTree

def print_time(message):
    d = datetime.datetime.today()
    print '%s/%s/%s %s:%s:%s %s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, message)    
    
def save_etrims(is08=True):
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/' if is08 else '04_etrims-ds/'
    #print an_list, im_list

    
    # annotations part
    path = root_path + an_name + et_name
    dir_list = os.listdir(path)
    signal = []
    print "annotations..."
    for file_name in dir_list:
        print file_name
        image_path = path + file_name
        image = Image.open(image_path)

        # get width and height
        w, h = image.size
        image_signal = [[0] * w] * h
        for i in xrange(w * h):
            x, y = i % w, i / w
            image_signal[y][x] = image.getpixel((x, y))
        signal.append(image_signal)
    
        
    # images part
    path = root_path + im_name + et_name
    dir_list = os.listdir(path)
    data = []
    print "images..."
    for file_name in dir_list:
        print file_name
        # list of cropped data
        image_path = path + file_name
        image = Image.open(image_path)

        # get width and height
        w, h = image.size
        image_data = [[0] * w] * h
        for i in xrange(w * h):
            x, y = i % w, i / w
            image_data[y][x] = list(image.getpixel((x,y)))
        data.append(image_data)

    dir_name = 'etrims_mat/'
    file_name = 'etrims_08.mat' if is08 else 'etrims_04.mat'

    mat = {'signal': signal, 'data': data, 'name': dir_list}
    scipy.io.savemat(dir_name + file_name, mat)
    print "save mat: done"

def load_etrims(is08=True, size=6, shuffle=True, OVERWRITE=False, visualize=True):
    print "Searching",
    dir_name = 'etrims_mat/'
    file_name = 'etrims_08.mat' if is08 else 'etrims_04.mat'
    path = dir_name + file_name
    print path, "..."

    if not os.path.exists(path):
        print path, "does not exist"
        print "making mat data"
        if not os.path.exists(dir_name):
            os.system('mkdir ' + dir_name)
        save_etrims(is08)
    elif OVERWRITE:
        print "OVERWRITE is True"
        print "making forcely mat data"
        save_etrims(is08)

    # scipy.io.load mat file
    print "load mat file"    
    mat = scipy.io.loadmat(path)
    mat_signal = mat['signal']
    mat_data = mat['data']

    # train index
    train_index = []
    DATA_SIZE = 60
    TRAIN_SIZE = 40
    if shuffle:
        # shuffle train index
        if DATA_SIZE < TRAIN_SIZE:
            raise Exception('DATA_SIZE < TRAIN_SIZE')
        while len(train_index) < TRAIN_SIZE:
            tmp = np.random.randint(DATA_SIZE)
            if not tmp in train_index:
                train_index.append(tmp)
    else:
        # unshuffle train index
        train_index = range(TRAIN_SIZE)

    # divide train and test
    test_or_train = [[], []]
    train_data, train_signal, test_data, test_signal = [],[],[],[]
    for i in xrange(DATA_SIZE):
        signal_w_h = mat_signal[0,i]
        data_w_h = mat_data[0,i]
        width = len(signal_w_h) - 1
        height = len(signal_w_h[0]) - 1

        signal_set = [test_signal, train_signal]
        data_set   = [test_data,   train_data]
        index = i in train_index

        # information
        test_or_train[index].append(i)

        # visualize
        if visualize:
            sys.stdout.write("\r %dth data train:%d test:%d labeling %.1f%%" % (i, len(test_or_train[1]), len(test_or_train[0]), (100.*i/DATA_SIZE)))
            sys.stdout.flush()

        for x,row in enumerate(signal_w_h):    
            for y,element in enumerate(row):
                # signal append
                signal = element
                signal_set[index].append(signal)

                # data append
                x1, x2 = max(0, x-size), min(width, x+size)
                y1, y2 = max(0, y-size), min(height, y+size)
                crop = data_w_h[x1:x2][y1:y2]
                data = [[0,0,0] * (2*size+1)] * (2*size+1)
                data_x1, data_x2 = x1-(x-size), 2*width-((x+size)-x2)
                data_y1, data_y2 = y1-(y-size), 2*height-((y+size)-y2)
                data[data_x1:data_x2][data_y1:data_y2] = crop
                data = [flatten for inner in data for flatten in inner]
                data = [flatten for inner in data for flatten in inner]
                data_set[index].append(data)

    print " done."
    print "[test, train] =", test_or_train
    return train_data, train_signal, test_data, test_signal


def etrims_tree(n_hidden = [1000], coef = [1000.], size=6):
    print_time('tree2etrims test size is %d' % size)
    print_time('load_etrims')
    train_data, train_signal, test_data, test_signal = load_etrims(size=size)

    num_function = 50
    print_time('train_DecisionTree num function is %d' % num_function)
    dt = DecisionTree(num_function=num_function)
    dt.fit(train_data, train_signal)

    print_time('test_DecisionTree')
    score = dt.score(test_data, test_signal)
    print_time('score is %f' % score)

    print_time('DecisionTree info')
    dt.info()


    elm_hidden = [(2*size+1)*(2*size+1)*2]
    print_time('train_ExtremeDecisionTree elm_hidden is %d, num function is %d' % (elm_hidden[0], num_function))
    edt = ExtremeDecisionTree(elm_hidden=elm_hidden, elm_coef=None, num_function=num_function)
    edt.fit(train_data, train_signal)

    print_time('test_ExtremeDecisionTree')
    score = edt.score(test_data, test_signal)
    print_time('score is %f' % score)

    print_time('test_ExtremeDecisionTree')
    score = edt.score(test_data, test_signal)
    print_time('score is %f' % score)

    print_time('ExtremeDecisionTree info')
    edt.info()

    print_time('tree2etrims test is finished !')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        size = 6
        print "############ warning: size is forcely", size, '#############'
    else:
        size = int(sys.argv[1])
    etrims_tree(size=size)
    
