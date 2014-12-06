# coding: utf-8

import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    return np.exp(sigmoid(x)) / np.exp(sigmoid(x)).sum()
    #return np.exp(x) / np.exp(x).sum()

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2
        
class RecurrentLayer(object):
    # 3 layers (input, hidden, output)
    def __init__(self, n_layer=None,
                 low=-1., high=1., alpha=0.1, beta=0.00001, trun=2,
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

    def train(self, input, teacher):
        self.get_output(input)
        self.back_propagation(teacher)

    def predict(self, input):
        self.hidden = None
        output = self.get_output(input)
        self.hidden = None
        return output
    
    def reset(self):
        self.input = None
        self.hidden = None
        self.past_data = []

  
def dataGenerator(n_input, n_output, size=20, term=2, seed=123):
    data = []
    np_rng = np.random.RandomState(seed)
    p = 0.5

    bino = np_rng.binomial(n=1, p=p, size=[size, n_input])
    for i in xrange(size):
        input = bino[i]
        head = max(0, i - term + 1)
        tail = i+1
        index = bino[head:tail].sum() % n_output
        label = [0] * n_output
        label[index] = 1
        data.append([input.tolist(), label])

    return np.array(data)


def test(epoch=1000):
    n_input  = 20
    n_hidden = 8
    n_output = 20
    train_size = 1000
    test_size  = 1000
    term = 1
    
    nn = RecurrentLayer(n_layer = [n_input, n_hidden, n_output])
    trainData = dataGenerator(n_input, n_output, train_size, term, 123)
    testData  = dataGenerator(n_input, n_output, test_size,  term, 1234)


    print "train ..."
    for i in xrange(epoch):
    #print "."
        for d in trainData:
            nn.train(d[0], d[1])
    print "done ."

    print "train ..."
    count = 0
    for d in trainData:
        output = nn.get_output(d[0])
        if output.argmax() == d[1].argmax():
            count += 1
        #print "input:",d[0], "output:",nn.get_output(d[0]), "signal",d[1]
    print "done ."
    print "score : ", count * 1./ train_size
    
    print "test ..."
    count = 0
    for d in testData:
        output = nn.get_output(d[0])
        if output.argmax() == d[1].argmax():
            count += 1
        #print "input:",d[0], "output:",nn.get_output(d[0]), "signal",d[1]
    print "done ."
    print "score : ", count * 1./ test_size
    
        
def pretest(epoch=1000):
    nn = RecurrentLayer(n_layer = [2, 10, 2])
    data = np.array([[[0,0],[0,1]],
                     [[0,1],[1,0]],
                     [[1,0],[0,1]],
                     [[1,1],[1,0]],
                     [[0,1],[1,0]],
                     [[0,0],[1,0]],
                     [[1,1],[0,1]],
                     [[1,0],[1,0]],
                     [[1,0],[0,1]],
                     [[1,1],[1,0]],
                     [[0,0],[0,1]],
                     [[0,1],[1,0]]])
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

    
