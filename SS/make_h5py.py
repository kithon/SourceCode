# -*- coding: utf-8 -*-
import os
import h5py
import random
import slic as sc
import numpy as np
import collections
from PIL import Image
    
##########################################################
##  Pic
##########################################################

class Pic(object):
    __slots__ = ['name', 'data', 'signal', 'spixel', 'palette',
                 'slength', 'scenter', 'sdic', 'w', 'h']
    def __init__(self, name, data, signal, spixel):
        self.name = name
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

    def getName(self):
        return self.name
        
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

    def getSCenter(self):
        return self.scenter
        
##########################################################
##  load_etrims
##########################################################
    
def load_etrims(radius, size, shuffle, n_superpixels, compactness):
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
        picture = Pic(file_name, image, annotation, spixel)
        test_or_train[index].append(picture)

    # ----- finish -----
    print "eTRIMS: train=%d test=%d" % (len(train_set), len(test_set))
    return train_set, test_set

def save_h5py(train_pic, test_pic, output_file, radius):
    # save picture detail to h5 file named $(file_name)
    h5file = h5py.File(output_file, 'w')
    
    # create optional information
    option_dir = 'option'
    h5file.create_group(option_dir)
    h5file.create_dataset(option_dir + '/num', data=len(train_pic)+len(test_pic))
    h5file.create_dataset(option_dir + '/radius', data=radius)
    h5file.create_dataset(option_dir + '/palette', data=test_pic[0].getPalette())

    print "create train test group"
    for dir in ['train', 'test']:
        h5file.create_group(dir)

    for pic,type in zip([train_pic, test_pic], ['train', 'test']):
        h5file.create_dataset(type + '/num', data=len(pic))
        for i,p in enumerate(pic):
            print type, ":", i
            dir = type + '/' + str(i) 
            h5file.create_group(dir)
            h5file.create_dataset(dir + '/name', data=p.getName())
            h5file.create_dataset(dir + '/size', data=p.getSize())
            h5file.create_dataset(dir + '/ssize', data=p.getSSize())
            h5file.create_dataset(dir + '/scenter', data=p.getSCenter())
            
            w,h = p.getSize()
            for x in xrange(w):
                print type, ":", i, ":", x
                x_dir = dir + '/' + str(x)
                h5file.create_group(x_dir)
                for y in xrange(h):
                    y_dir = x_dir + '/' + str(y)
                    h5file.create_group(y_dir)
                    h5file.create_dataset(y_dir + '/data', data=p.getData(x,y))
                    h5file.create_dataset(y_dir + '/signal', data=p.getSignal(x,y))
                    h5file.create_dataset(y_dir + '/crop', data=p.cropData(x,y,radius))
                    h5file.create_dataset(y_dir + '/sindex', data=p.getSIndex(x,y))
                h5file.flush()
                
    h5file.flush()
    h5file.close()
    
##########################################################
##  Main Function
##########################################################

def main(boxSize, dataSize, unShuffle, num_superpixels, compactness, file_name='pic_set.h5'):
    # ----- get radius -----
    radius = (boxSize - 1) / 2

    # ----- get picture -----
    train_pic, test_pic = load_etrims(radius=radius, size=dataSize, shuffle=not unShuffle, n_superpixels=num_superpixels, compactness=compactness)

    # ----- save -----
    save_h5py(train_pic, test_pic, file_name, radius)

    # ----- finish -----
    print 'eTRIMS: finish'

    
