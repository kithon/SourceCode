# coding: utf-8
import os
import gzip
import cPickle
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer

def load_mnist():
    f = gzip.open('../Dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def mnist_dec():
    # initialize
    train_set, valid_set, test_set = load_mnist()
    train_data, train_target = train_set
    valid_data, valid_target = valid_set
    test_data, test_target = test_set
    #train_target = LabelBinarizer().fit_transform(train_target)
    #valid_target = LabelBinarizer().fit_transform(valid_target)
    #test_target = LabelBinarizer().fit_transform(test_target)
    
    # size
    train_size = 50000 # max 50000
    valid_size = 10000 # max 10000
    test_size = 10000 # max 10000
    train_data, train_target = train_data[:train_size], train_target[:train_size]
    test_data, test_target = test_data[:test_size], test_target[:test_size]
    valid_data, valid_target = valid_data[:valid_size], valid_target[:valid_size]

    # model
    model = DecisionTreeClassifier()

    # fit
    print "fitting ..."
    model.fit(train_data, train_target)

    # test
    print "test score is ",
    score = model.score(test_data, test_target)
    print score

    # valid
    print "valid score is ",
    score = model.score(valid_data, valid_target)
    print score

    # export_graphviz(model)
    export_graphviz(model)

if __name__ == "__main__":
    mnist_dec()
