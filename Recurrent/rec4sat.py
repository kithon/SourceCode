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
    num = 100
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

    # generate sat data
    sat = []
    for i in xrange(num):
        data, signal = generateSat(n_len, n_val, n_sat)
        sat.append([data, signal])
        if not checkSat(data, signal, n_val):
            print "error !!"
            print data, signal

    # generate model
    nn = RecurrentLayer(n_layer = [2 * n_val + 1, 20, n_val], trun=n_len, a_output = sigmoid)
    #nn = RecurrentLayer(n_layer = [2 * n_val + 1, 20, n_val], trun=n_len)

    # test for train data
    count = 0
    for data in sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        #print output
        if checkSat(data[0], output, n_val):
            count += 1
    print "accuracy:", count,  "/", num
    
    # train
    for i in xrange(epoch):
        print "epoch:", i
        for data in sat:
            for d in data[0]:
                nn.get_output(d)
                #print nn.get_output(d)
            signal = data[1]
            
            nn.back_propagation(signal)
            nn.reset()
            
        # test for train data
        count = 0
        for data in sat:
            for d in data[0]:
                #output = nn.get_output(d)
                output = sign(nn.get_output(d))
            nn.reset()
            #print output
            if checkSat(data[0], output, n_val):
                count += 1
        print "accuracy:", count,  "/", num

    
    # test for train data
    count = 0
    for data in sat:
        for d in data[0]:
            #output = nn.get_output(d)
            output = sign(nn.get_output(d))
        nn.reset()
        if checkSat(data[0], output, n_val):
            count += 1
    print "accuracy:", count,  "/", num

if __name__ == '__main__':
    testRecurrent()
