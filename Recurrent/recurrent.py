# coding: utf-8

import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2
        
class RecurrentLayer(object):
    # 3 layers (input, hidden, output)
    def __init__(self, n_layer=None,
                 low=-1., high=1., alpha=0.5, beta=0.01, trun=3,
                 a_hidden=sigmoid, a_output=softmax, np_rng=None):
        # parameter
        if n_layer is None:
            raise Exception("set layer size")

        self.input = None
        self.hidden = None
        self.past_data = []
        self.n_input, self.n_hidden, self.n_output = n_layer
        self.alpha = alpha
        self.beta = beta
        self.trun = trun
        self.a_hidden = a_hidden
        self.a_output = a_output

        if np_rng is None:
            np_rng = np.random.RandomState(123)
        self.np_rng = np_rng

        # each weight
        self.u = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input, self.n_hidden))
        self.w = self.np_rng.uniform(low = low, 
                                     high = high,
                                     size = (self.n_hidden, self.n_hidden))
        self.v = self.np_rng.uniform(low = low, 
                                     high = high,
                                     size = (self.n_hidden, self.n_output))

                                     
    def get_output(self, input):
        past = []
        # append past_input
        if not self.input is None:
            past.append(self.input)
                
        # set input
        self.input = input

        # append past_hidden
        if not self.hidden is None:
            past.append(self.hidden)

        # set hidden
        if self.hidden is None:
            self.hidden = np.zeros(self.n_hidden)
            
        self.hidden = self.a_hidden(np.dot(self.u.T, self.input) +
                                    np.dot(self.w.T, self.hidden))
        
        # set output
        self.output = self.a_output(np.dot(self.v.T, self.hidden))
        return self.output

        # append past_data
        if len(past) != 0:
            self.past_data.append(past)
            if len(self.past_data) > self.trun:
                self.past_data.pop(0)

    def back_propagation(self, teacher):
        # back propagation through time
        
        # update self.v
        e0 = np.array([teacher - self.output])
        self.v = (self.v +
                  self.alpha *
                  np.dot(np.array([self.hidden]).T, e0) -
                  self.beta * self.v)


        # update self.u self.w
        error = (np.dot(e0, self.v.T) *
                 np.array(self.hidden * (1. - self.hidden)))
        sigma_u = np.dot(np.array([self.input]).T, error)
        sigma_w = 0
        
        for data in self.past_data[::-1]:
            input, hidden = data
            error = (np.dot(error, self.w.T) * np.array(hidden * (1. - hidden)))
            print "input", np.array([input]).T.shape
            sigma_u += np.dot(np.array([input]).T, error)
        
        self.u = (self.u + self.alpha * sigma_u - self.beta * self.u)
        
        # update self.w
                 
  

def test(epoch=10000):
    nn = RecurrentLayer(n_layer = [2, 4, 2])
    data = np.array([[[0,0],[0,1]],
                     [[0,1],[1,0]],
                     [[1,0],[1,0]],
                     [[1,1],[0,1]]])
    #data = np.array([[[1,1], [0,1]]])
    
    for i in xrange(epoch):
        print i, "th train"
        for d in data:
            print d[0], nn.get_output(d[0]), d[1]
            nn.back_propagation(d[1])

if __name__ == "__main__":
    print "test"
    test()

    
