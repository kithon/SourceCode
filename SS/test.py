# -*- coding: utf-8 -*-
import test_forest

# config Tree Type
isREG = True
isELMREG = True

# config forest
n_estimator = 2
max_depth = 3
sample_pertree = 0.75
sample_freq = 5
max_features = 10
min_leaf_nodes = None
alpha = 0.8
learning_rate = 0.9
verpose = None

# config fileName
input_file = 'data/pic_set_mini.h5'
result_dir = 'result'
log_file = "test.log"

if __name__ == '__main__':
    test_forest.do_forest(isREG, isELMREG,
                          n_estimator, max_depth, sample_pertree, sample_freq,
                          max_features, min_leaf_nodes, alpha, learning_rate, verpose,
                          input_file, result_dir, result_dir + '/' + log_file)
