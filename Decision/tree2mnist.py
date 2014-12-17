# coding: utf-8
import gzip
import cPickle
from tree import DecisionTree, ExtremeDecisionTree

def load_mnist():
    f = gzip.open('../Dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def mnist_mlelm(n_hidden=[1000]):
    print "hidden:", n_hidden

    # initialize
    train_set, valid_set, test_set = load_mnist()
    train_data, train_target = train_set
    valid_data, valid_target = valid_set
    test_data, test_target = test_set
    
    # size
    train_size = 500 # max 50000
    valid_size = 100 # max 10000
    test_size = 100 # max 10000

    train_data, train_target = train_data[:train_size], train_target[:train_size]
    valid_data, valid_target = valid_data[:valid_size], valid_target[:valid_size]
    test_data, test_target = test_data[:test_size], test_target[:test_size]

    # add valid_data/target to train_data/target
    """
    train_data   = train_data   + valid_data
    train_target = train_target + valid_target
    """

    # model
    dt = DecisionTree()
    edt1 = ExtremeDecisionTree(elm_hidden=n_hidden)
    edt2 = ExtremeDecisionTree(elm_hidden=n_hidden, elm_coef=[1000., 100., 1000.])

    # fit
    #print "fitting ..."
    dt.fit(train_data, train_target)
    edt1.fit(train_data, train_target)
    edt2.fit(train_data, train_target)

    # test
    print "test score is ",
    score_dt = dt.score(test_data, test_target)
    score_edt1 = edt1.score(test_data, test_target)
    score_edt2 = edt2.score(test_data, test_target)
    print score_dt, score_edt1, score_edt2


if __name__ == "__main__":
    mnist_mlelm([700, 700, 15000])
    #mnist_mlelm([700, 700, 15000])
