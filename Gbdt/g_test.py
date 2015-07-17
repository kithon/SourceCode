# -*- coding: utf-8 -*-
import gradient_boosting

# config data
boxSize = 15# 15
dataSize = 6
unShuffle = False

# config Tree Type
isREG = False
isELMREG = True

# config forest
n_estimator = 10
max_depth = 10
sample_pertree = 0.75
sample_freq = 5
max_features = 10
min_leaf_nodes = None
alpha = 0.8
learning_rate = 0.9
verpose = None

# config REG
reg_args = {'radius': (boxSize - 1) / 2}

# config ELMREG
reg_args = {'radius': (boxSize - 1) / 2,
            'sample_size': boxSize * boxSize * 3,
            'elm_hidden': boxSize * boxSize * 3 * 2}

# config slic
n_superpixels = 500
compactness = 10

# config fileName
fileName = "g_test.log"

if __name__ == '__main__':
    gradient_boosting.do_forest(boxSize, dataSize, unShuffle, 
                                isREG, isELMREG,
                                n_estimator, max_depth, sample_pertree, sample_freq,
                                max_features, min_leaf_nodes, alpha, learning_rate, verpose,
                                reg_args, elmreg_args,
                                n_superpixels, compactness,
                                file_name):
