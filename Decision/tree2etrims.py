# -*- coding: utf-8 -*-
import os
import sys
import datetime
import numpy as np
from PIL import Image
from tree import DecisionTree, ExtremeDecisionTree

def print_time(message):
    d = datetime.datetime.today()
    print '%s/%s/%s %s:%s:%s.%s %s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, message)    
    
def load_etrims(is08=True, size=6, shuffle=True, visualize=True):
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/' if is08 else '04_etrims-ds/'
    #print an_list, im_list

    
    # annotations part
    path = root_path + an_name + et_name
    dir_list = os.listdir(path)
    mat_signal = []

    print "annotations..."
    for file_name in dir_list:
        print file_name
        image_path = path + file_name
        image = Image.open(image_path)

        # get width and height
        w, h = image.size
        image_signal = []
        for y in xrange(h):
            row = []
            for x in xrange(w):
                row.append(image.getpixel((x, y)))
            image_signal.append(row)
        mat_signal.append(image_signal)

        
    # images part
    path = root_path + im_name + et_name
    dir_list = os.listdir(path)
    mat_data = []
    print "images..."
    for file_name in dir_list:
        print file_name
        # list of cropped data
        image_path = path + file_name
        image = Image.open(image_path)

        # get width and height
        w, h = image.size
        image_data = []
        for y in xrange(h):
            row = []
            for x in xrange(w):
                row.append(list(image.getpixel((x,y))))
            image_data.append(row)
        mat_data.append(image_data)
        

        
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
    for iteration in xrange(DATA_SIZE):
        signal_w_h = mat_signal[iteration]
        data_w_h = mat_data[iteration]
        width = len(signal_w_h) - 1
        height = len(signal_w_h[0]) - 1

        signal_set = [test_signal, train_signal]
        data_set   = [test_data,   train_data]
        index = iteration in train_index

        # information
        test_or_train[index].append(iteration)

        # visualize
        if visualize:
            sys.stdout.write("\r %dth data train:%d test:%d labeling %.1f%%" % (iteration, len(test_or_train[1]), len(test_or_train[0]), (100.*iteration/DATA_SIZE)))
            sys.stdout.flush()

        for x,row in enumerate(signal_w_h):    
            for y,element in enumerate(row):
                #print x,y
                # signal append
                signal = element
                signal_set[index].append(signal)

                # data append
                x1, x2 = max(0, x-size), min(width, x+(size+1))
                y1, y2 = max(0, y-size), min(height, y+(size+1))
                crop = data_w_h[x1:x2]

                temp = []
                for i in xrange(len(crop)):
                    temp.append(crop[i][y1:y2])
                crop = temp

                data = []
                for i in xrange(2*size+1):
                    row = []
                    for j in xrange(2*size+1):
                        row.append([0,0,0])
                    data.append(row)                
                
                data_x1, data_y1 = x1-(x-size), y1-(y-size)
                data_x2 = data_x1 + x2 - x1
                data_y2 = data_y1 + y2 - y1
                for i in xrange(data_x1, data_x2):
                    for j in xrange(data_y1, data_y2):
                        if iteration == 1:
                            print i,j,len(crop),len(crop[0]), data_x1, data_x2, data_y1, data_y2, x1, x2, y1, y2
                        data[i][j] = crop[i-data_x1][j-data_y1]

                #print data, len(data), len(data[0])
                    
                data = [flatten for inner in data for flatten in inner]
                data = [flatten for inner in data for flatten in inner]
                #print "one", data
                data_set[index].append(data)

    print " done."
    print "[test, train] =", test_or_train
    return train_data, train_signal, test_data, test_signal


def etrims_tree(n_hidden = [1000], coef = [1000.], size=6):
    print_time('tree2etrims test size is %d' % size)
    print_time('load_etrims')
    train_data, train_signal, test_data, test_signal = load_etrims(size=size)

    num_function = 100
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
    
