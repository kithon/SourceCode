# -*- coding: utf-8 -*-
import sys
import make_h5py

# config data
boxSize = 15# 15
dataSize = 3
unShuffle = False

# config slic
n_superpixels = 500
compactness = 10

# config fileName
if len(sys.argv) > 1:
    dataSize = 60
    file_name = sys.argv[1]
else:
    file_name = "pic_set_mini.h5"
    
#file_name = "pic_set_mini.h5"
#file_name = "data/pic_set1.h5"
#file_name = "data/pic_set2.h5"
#file_name = "data/pic_set3.h5"
#file_name = "data/pic_set4.h5"
#file_name = "data/pic_set5.h5"

print "write data to", file_name
if __name__ == '__main__':
    make_h5py.main(boxSize, dataSize, unShuffle, n_superpixels, compactness, file_name)
