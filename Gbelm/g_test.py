# -*- coding: utf-8 -*-
import gradient_elm

# config data
boxSize = 25# 15
dataSize = 6
unShuffle = False

# config forest
n_estimator = 10
sample_perelm = 0.05
sample_freq = 5
elm_hidden = boxSize * boxSize * 3 * 2
learning_rate = 0.9

# config slic
n_superpixels = 500
compactness = 10

# config fileName
file_name = "g_test.log"

if __name__ == '__main__':
    gradient_elm.do_elm(boxSize, dataSize, unShuffle, 
                        n_estimator, sample_perelm, sample_freq,
                        elm_hidden, learning_rate, 
                        n_superpixels, compactness,
                        file_name)
