# -*- coding: utf-8 -*-
import os
import random
import argparse
from PIL import Image
from etrims_tree import DecisionTree, ExtremeDecisionTree, BinaryExtremeDecisionTree, Pic, print_time, print_parameter, change
    
##########################################################
##  load_etrims
##########################################################
    
def load_etrims(radius, size, is08, shuffle):
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
        print_time("eTRIMS: %s" % file_name)

    # ----- finish -----
    print_parameter(train_index)
    print_time("eTRIMS: train=%d test=%d" % (len(train_set), len(test_set)))
    return train_set, test_set


def etrims_tree(radius, size, d_limit, remove, unshuffle, four, num, parameter, t_args):
    # ----- initialize -----
    print_parameter([radius, size, d_limit, unshuffle, four, num, t_args])
    print_time('eTRIMS: radius=%d, depth_limit=%s, data_size=%d, num_func=%d' % (radius, str(d_limit), size, num))
    print_time('eTRIMS: load')
    train_set, test_set = load_etrims(radius=radius, size=size, is08=not four, shuffle=not unshuffle)
    isDT, isEDT, isBEDT = t_args
    
    # ----- Decision Tree -----
    if isDT:
        print_time('DecisionTree (overlap): init')
        dt = DecisionTree(radius=radius, num_function=num, remove=remove)
        
        print_time('DecisionTree (overlap): train')
        dt.fit(train_set, test_set, d_limit=d_limit, overlap=True)
        
        print_time('DecisionTree (overlap): info')
        dt.info()

        print_time('DecisionTree: init')
        dt = DecisionTree(radius=radius, num_function=num, remove=remove)
        
        print_time('DecisionTree: train')
        dt.fit(train_set, test_set, d_limit=d_limit, overlap=False)
        
        print_time('DecisionTree: info')
        dt.info()
    

    # ----- Extreme Decision Tree -----
    if isEDT:
        print_time('ExtremeDecisionTree: init')
        edt = ExtremeDecisionTree(radius=radius, num_function=num, remove=remove)
        
        print_time('ExtremeDecisionTree: train')
        edt.fit(train_set, test_set, d_limit=d_limit, overlap=False)
        
        print_time('ExtremeDecisionTree: info')
        edt.info()

    # ----- Binary Extreme Decision Tree -----
    if isBEDT:
        print_time('BinaryExtremeDecisionTree: init')
        bedt = BinaryExtremeDecisionTree(radius=radius, num_function=num, remove=remove)
        
        print_time('BinaryExtremeDecisionTree: train')
        bedt.fit(train_set, test_set, d_limit=d_limit, overlap=False)
                
        print_time('BinaryExtremeDecisionTree: info')
        bedt.info()


    # ----- finish -----
    print_time('eTRIMS: finish')


if __name__ == '__main__':
    # ----- parser description -----
    parser = argparse.ArgumentParser(description='Test eTRIMS-08 Segmentation Dataset (need etrims_tree.py)')
    parser.add_argument("name", type=str, default='result.log', help="set file name")
    parser.add_argument("radius", type=int, default=2, nargs='?', help="set image radius")
    parser.add_argument("size", type=int, default=60, nargs='?', help="set data size")
    parser.add_argument("limit", type=int, nargs='?', help="set depth limit")
    parser.add_argument("-r", "--removeparam", action='store_true',  help="remove parameter")
    parser.add_argument("-u", "--unshuffle", action='store_true',  help="not shuffle dataset")
    parser.add_argument("-f", "--four", action='store_true',  help="use eTRIMS-04 dataset")
    parser.add_argument("-n", "--num", metavar="num", type=int, default=5,  help="set number of function")
    parser.add_argument("-p", "--parameter", metavar='file', type=str, help="set trained parameter")
    parser.add_argument("-t", "--tree", metavar='{d,e,b}', default='deb', help="run tree individually")
    
    # ----- etrims_tree -----
    args = parser.parse_args()
    change(args.name)
    
    t_args = map(lambda x:x in args.tree, ['d','e','b'])
    if True in t_args:
        etrims_tree(radius=args.radius, size=args.size, d_limit=args.limit, remove=args.removeparam, unshuffle=args.unshuffle,
                    four=args.four, num=args.num, parameter=args.parameter, t_args=t_args)
    else:
        print_time('etrims_test.py: error: argument -t/--tree: expected {d,e,b} argument')

