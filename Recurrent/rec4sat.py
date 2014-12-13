# -*- coding: utf-8 -*-
import random
import numpy as np
from recurrent import RecurrentLayer, sigmoid

def biran(size):
    return int(random.random() * size)

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

def generateSat(n_len, n_val, n_sat):
    signal = []
    for i in xrange(n_val):
        signal.append(biran(2))

    input = []
    for i in xrange(n_len):
        index = []
        in_sig = biran(n_val)
        while len(index) < n_sat - 1:
            temp = biran(n_val)
            if not temp in index and temp != in_sig:
                index.append(temp)

        clause = [0] * (2 * n_val + 1)
        clause[in_sig + (1 - signal[in_sig]) * n_val] = 1
        for temp in index:
            neg = biran(2)
            clause[temp + neg * n_val] = 1
        input.append(clause)

    return input, signal    

def checkSat(input, output, n_val):
    for sat in input:
        isClause = False
        for i in xrange(n_val):
            if (sat[i] == 1 and output[i] == 1) or (sat[i+n_val] == 1 and output[i] == 0):
                isClause = True
        if not isClause:
            return False
    return True

def testRecurrent():
    # initialize
    epoch = 1000
    train_num = 500
    test_num = 100
    n_sat = 3
    n_val = 10
    n_len = 30

    """
    count = 0
    for i in xrange(num):
        data, signal = generateSat(n_len, n_val, n_sat)
         #print data
        if checkSat(data, signal, n_val):
            count += 1
    print count
    """

    # generate train sat data
    train_sat = []
    for i in xrange(train_num):
        data, signal = generateSat(n_len, n_val, n_sat)
        train_sat.append([data, signal])
        if not checkSat(data, signal, n_val):
            print "train error !!"
            print data, signal

    # generate test sat data
    test_sat = []
    for i in xrange(test_num):
        data, signal = generateSat(n_len, n_val, n_sat)
        test_sat.append([data, signal])
        if not checkSat(data, signal, n_val):
            print "test error !!"
            print data, signal

    # generate model
    nn = RecurrentLayer(n_layer = [2 * n_val + 1, 20, n_val], alpha=0.005, beta=0.000001, trun=n_len, a_output = sigmoid)
    #nn = RecurrentLayer(n_layer = [2 * n_val + 1, 20, n_val], trun=n_len)

    # train acuraccy
    count = 0
    for data in train_sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        #print output
        if checkSat(data[0], output, n_val):
            count += 1
    print "train accuracy:", count,  "/", train_num

    # test acuraccy
    count = 0
    for data in test_sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        #print output
        if checkSat(data[0], output, n_val):
            count += 1
    print "test accuracy:", count,  "/", test_num
    
    # train
    for i in xrange(epoch):
        print "epoch:", i
        for data in train_sat:
            for d in data[0]:
                nn.get_output(d)
                #print nn.get_output(d)
            signal = data[1]
            
            nn.back_propagation(signal)
            nn.reset()

        # train acuraccy
        count = 0
        for data in train_sat:
            for d in data[0]:
                #output = nn.get_output(d)
                output = sign(nn.get_output(d))
            nn.reset()
            #print output
            if checkSat(data[0], output, n_val):
                count += 1
        print "train accuracy:", count,  "/", train_num

        # test acuraccy
        count = 0
        for data in test_sat:
            for d in data[0]:
                #output = nn.get_output(d)
                output = sign(nn.get_output(d))
            nn.reset()
            #print output
            if checkSat(data[0], output, n_val):
                count += 1
        print "test accuracy:", count,  "/", test_num
    

    # train acuraccy
    count = 0
    for data in train_sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        #print output
        if checkSat(data[0], output, n_val):
            count += 1
    print "train accuracy:", count,  "/", train_num

    # test acuraccy
    count = 0
    for data in test_sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        #print output
        if checkSat(data[0], output, n_val):
            count += 1
    print "test accuracy:", count,  "/", test_num
    

if __name__ == '__main__':
    testRecurrent()
