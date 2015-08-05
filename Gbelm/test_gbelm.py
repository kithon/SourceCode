# -*- coding: utf-8 -*-
import random
import numpy as np
from extreme import ELMRegressor


def softmax(data, t=1.0):
    # softmax element-wise
    e = np.exp(np.array(data) / t)
    dist = e / np.sum(e)
    return dist

def create_data(input_size, label_size, num_data):
    input_list, signal_list = [], []
    ident = np.identity(label_size)
    for i in xrange(num_data):
        label = random.randint(0, label_size - 1)
        input_list.append([random.random() for col in xrange(input_size)])
        signal_list.append(ident[label])
    return input_list, signal_list
        
def test_gbelm():
    input_size = 15 * 15 * 3 # 100
    hidden_size = 15 * 15 * 3 * 2 # 200
    label_size = 8 # 8
    num_data = 100
    input_list, signal_list = create_data(input_size, label_size, num_data)
    output_list = [[1 for row in xrange(label_size)] for col in xrange(num_data)]

    elm_list = []
    n_estimator = 1000
    data_perelm = 0.25
    sample_size = int(num_data * data_perelm)

    lr = 0.9
    for est in xrange(n_estimator):        
        elm = ELMRegressor(n_hidden=hidden_size)        
        sample_index = random.sample(range(num_data), sample_size)
        input  = [input_list[index] for index in sample_index]
        signal = [signal_list[index] for index in sample_index]
        output = [output_list[index] for index in sample_index]
        gradient = [(np.array(s) - np.array(softmax(o))).tolist() for s,o in zip(signal, output)]

        
        elm.fit(input, gradient)
        elm_list.append(elm)

        for index in xrange(num_data):
            output_list[index] = (np.array(output_list[index]) + lr * np.array(elm.predict(input_list[index]))).tolist()

        print "Learning rate:", lr, "Gradient:", np.mean(gradient), np.var(gradient), np.max(gradient), np.min(gradient)
        #print [softmax(o) for o in output_list]
        if np.var(gradient):
            lr *= 1.1#0.99995

if __name__ == '__main__':
    test_gbelm()

