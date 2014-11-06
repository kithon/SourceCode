# coding: utf-8
import gzip
import cPickle
from extreme import ELMClassifier

def load_mnist():
    f = gzip.open('../Dataset/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def mnist_elm(n_hidden=50, domain=[-1., 1.]):
    print "hidden:", n_hidden

    # initialize
    train_set, valid_set, test_set = load_mnist()
    train_data, train_target = train_set
    valid_data, valid_target = valid_set
    test_data, test_target = test_set
    
    # size
    train_size = 50000 # max 50000
    valid_size = 10000 # max 10000
    test_size = 10000 # max 10000

    train_data, train_target = train_data[:train_size], train_target[:train_size]
    valid_data, valid_target = valid_data[:valid_size], valid_target[:valid_size]
    test_data, test_target = test_data[:test_size], test_target[:test_size]

    # model
    model = ELMClassifier(n_hidden = n_hidden, domain = domain)

    # fit
    #print "fitting ..."
    model.fit(train_data, train_target)

    # test
    print "test score is ",
    score = model.score(test_data, test_target)
    print score

    # valid
    print "valid score is ",
    score = model.score(valid_data, valid_target)
    print score


if __name__ == "__main__":
    mnist_elm(100, [-0.02, 0.02])
    mnist_elm(200, [-0.02, 0.02])
    mnist_elm(300, [-0.02, 0.02])
    mnist_elm(400, [-0.02, 0.02])
    mnist_elm(500, [-0.02, 0.02])
    mnist_elm(600, [-0.02, 0.02])
    mnist_elm(700, [-0.02, 0.02])
    mnist_elm(800, [-0.02, 0.02])
    mnist_elm(900, [-0.02, 0.02])
    mnist_elm(1000, [-0.02, 0.02])
