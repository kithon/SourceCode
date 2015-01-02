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
    
    # path
    an_path = root_path + an_name + et_name
    im_path = root_path + im_name + et_name
    dir_list = os.listdir(an_path)
    train_data, train_signal, test_data, test_signal = [],[],[],[]
    signal_set = [test_signal, train_signal]
    data_set   = [test_data,   train_data]
        
    print "image datas and annotations..."
    for i, dis in enumerate(dir_list):
        file_name = dis.split(".")[0]
        #print file_name
        annot_path = an_path + file_name + ".png"
        annotation = Image.open(annot_path)
        image_path = im_path + file_name + ".jpg"
        image = Image.open(image_path)

        # get width, height and index
        w, h = image.size
        index = i in train_index

        """
        ###### debug ####################################################
        index = True if i == 0 else False
        """

        for t in xrange(w * h):
            x, y = t % w, t / w
            
            # annotation
            signal_set[index].append(annotation.getpixel((x,y)))
            
            # data
            crop_data = []
            crop = image.crop([x-size, y-size, x+size, y+size]).getdata()
            for c in crop:
                # convert RGB to list
                crop_data += list(c)
            data_set[index].append(crop_data)

            # visualize
            if visualize and t % 500 == 0:
                sys.stdout.write("\r%s %.1f%%" % (file_name, (100.*(y*w+x)/(w*h))))
                sys.stdout.flush()

        # done
        if visualize:
            sys.stdout.write("\r")
            sys.stdout.flush()
            print file_name,  "done                  "

        """
        ###### debug #####################################################
        if i == 1:
            break
        """

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
    
