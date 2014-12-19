# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from tree import DecisionTree, ExtremeDecisionTree

def load_etrims(is08=True, size=2):
    root_path = '../Dataset/etrims-db_v1/'
    an_name = 'annotations/'
    im_name = 'images/'
    et_name = '08_etrims-ds/' if is08 else '04_etrims-ds/'
    #print an_list, im_list

    
    # annotations part
    path = root_path + an_name + et_name
    dir_list = os.listdir(path)
    signal = []
    for file_name in dir_list:
        print file_name
        image_signal = []
        image_path = path + file_name
        image = Image.open(image_path)
        
        # get width and height
        w, h = image.size
        for i in xrange(w * h):
            x, y = i % w, i / w
            image_signal.append(image.getpixel((x, y)))
        signal.append(image_signal)
   

    """
    # images part
    path = root_path + im_name + et_name
    dir_list = os.listdir(path)
    data = []
    for file_name in dir_list:
        print file_name
        # list of cropped data
        image_data = []
        image_path = path + file_name
        image = Image.open(image_path)

        # get width and height
        w, h = image.size
        for i in xrange(w * h):
            crop_data = []
            x, y = i % w, i / w
            crop = image.crop([x-size, y-size, x+size, y+size]).getdata()
            for c in crop:
                crop_data += list(c)
            image_data.append(data)
        data.append(image_data)
     """
    
    
def etrims_tree(n_hidden = [1000], coef = [1000.]):
    load_etrims()


if __name__ == '__main__':
    etrims_tree()
    
