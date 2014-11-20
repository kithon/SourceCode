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
                 low=-1., high=1., alpha=0.5, beta=0.0001, trun=3,
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
        if self.input is not None:
            past.append(self.input)
                
        # set input
        self.input = input

        # append past_hidden
        if self.hidden is not None:
            past.append(self.hidden)

        # set hidden
        if self.hidden is None:
            self.hidden = np.zeros(self.n_hidden)

        #print self.hidden
        self.hidden = self.a_hidden(np.dot(self.u.T, self.input) + 
                                    np.dot(self.w.T, self.hidden))
        
        # set output
        #print "dot", np.dot(self.v.T, self.hidden)
        self.output = self.a_output(np.dot(self.v.T, self.hidden))

        # append past_data (store (self.trun + 1) past's)
        if len(past) != 0:
            self.past_data.append(past)
            if len(self.past_data) > (self.trun + 1):
                self.past_data.pop(0)

        return self.output

    def back_propagation(self, teacher):
        # back propagation through time
        
        # update self.v
        e0 = np.array([teacher - self.output])
        self.v = ((1. - self.beta) * self.v +
                  self.alpha * np.dot(np.array([self.hidden]).T, e0))
        

        # update self.u self.w
        error = (np.dot(e0, self.v.T) *
                 np.array(self.hidden * (1. - self.hidden)))
        sigma_u = np.dot(np.array([self.input]).T, error)
        sigma_w = np.zeros(self.w.shape)

        if len(self.past_data) != 0:
            input, hidden = self.past_data[::-1][0]
            sigma_w += np.dot(np.array([hidden]).T, error)
            
        for i in xrange(len(self.past_data) - 1):
            input, hidden = self.past_data[::-1][i]
            p_input, p_hidden = self.past_data[::-1][i+1]
            
            error = (np.dot(error, self.w.T) * np.array(hidden * (1. - hidden)))

            sigma_u += np.dot(np.array([input]).T, error)
            sigma_w += np.dot(np.array([p_hidden]).T, error)

        self.u = (1. - self.beta) * self.u + self.alpha * sigma_u
        self.w = (1. - self.beta) * self.w + self.alpha * sigma_w
  

def test(epoch=500):
    nn = RecurrentLayer(n_layer = [2, 10, 2])
    data = np.array([[[0,0],[0,1]],
                     [[0,1],[1,0]],
                     [[1,0],[1,0]],
                     [[1,1],[0,1]]])
    #data = np.array([[[1,1], [0,1]]])
    
    for i in xrange(epoch):
        #print i, "th train"
        
        for d in data:
            #print nn.get_output(d[0]), "to",
            nn.get_output(d[0])
            nn.back_propagation(d[1])
            #print d[0], nn.get_output(d[0]), d[1]
            #nn.back_propagation(d[1])
            
    for d in data:
        print d[0], nn.get_output(d[0]), d[1]

            
if __name__ == "__main__":
    print "test"
    test()

    
