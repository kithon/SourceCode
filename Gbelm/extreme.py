# coding: utf-8
# online learning

import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2


##########################################################
##  Extreme Learning Machine Regressor
##########################################################
    
class ELMRegressor(object):
    """
    ELMRegressor : 
        __init__ : 
            activation : Layer's activation
            vector : Policy of generating Layers weight ('orthogonal' or 'random')
            n_hidden : Hidden Layer's number of neuron
            coef : coefficient for Layer's ridge redression
            seed : seed for np.random.RandomState
            domain : domain for initial value of weight and bias
    """

    def __init__(self, activation=sigmoid, vector='random',
                 n_hidden=50, coef=None, seed=123, domain=[-1., 1.]):
        self.activation = activation
        self.vector = vector
        self.coef = coef
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta

    def fit(self, input, teacher):
        # set input, teacher and class
        self.input = input
        self.teacher = teacher
        self.n_input = len(input[0])
        self.n_output = len(teacher[0])

        # weight and bias
        low, high = self.domain
        weight = self.np_rng.uniform(low = low,
                                     high = high,
                                     size = (self.n_input,
                                             self.n_hidden))
        bias = self.np_rng.uniform(low = low,
                                   high = high,
                                   size = self.n_hidden)

        # condition : orthogonal or random
        if self.vector == 'orthogonal':
            # orthogonalize weight
            for i in xrange(len(weight)):
                w = weight[i]
                for j in xrange(0,i):
                    w = w - weight[j].dot(w) * weight[j]
                w = w / np.linalg.norm(w)
                weight[i] = w

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                denom = bias / denom
                    
        elif self.vector == 'random':
            # regularize weight
            for i,w in enumerate(weight):
                denom = np.linalg.norm(w)
                if denom != 0:
                    weight[i] = w / denom

            # regularize bias
            denom = np.linalg.norm(bias)
            if denom != 0:
                bias = bias / denom
                    
        else:
            print "warning: vector isn't orthogonal or random"
            
        
        # generate self weight and bias
        self.weight = weight
        self.bias = bias
            
        # generate self layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.coef)

        # fit layer
        self.layer.fit(input, teacher)
        return self.get_weight(), self.get_bias(), self.get_beta()
        
    def predict_batch(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(o)        #print "outputs", predict_output

        return predict_output
   
    def predict(self, input):
        return self.predict_batch([input])[0]

    def score(self, input, teacher):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.predict_batch(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length
    
##########################################################
##  Layer
##########################################################
    
class Layer(object):
    """
    Layer : used for Extreme Learning Machine
        __init__ : 
            activation : activation from input to hidden
            n_{input, hidden, output} : each layer's number of neuron
            c : coefficient for ridge regression
            w : weight from input to hidden layer
            b : bias from input to hidden layer
            beta : beta from hidden to output layer
    """
    def __init__(self, activation, size, w, b, c, visualize=False):
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.c = c
        self.w = w
        self.b = b
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])
        self.visualize = visualize

    def get_beta(self):
        return self.beta
        
    def get_i2h(self, input):
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        hidden = self.get_i2h(input)  # from input to hidden
        output = self.get_h2o(hidden) # from hidden to output
        return output
    
    def fit(self, input, signal, alpha=0.0001):
        # get activation of hidden layer
        H = []
        for i, d in enumerate(input):
            if self.visualize:
                sys.stdout.write("\r    input %d" % (i+1))
                sys.stdout.flush()
            H.append(self.get_i2h(d))
        if self.visualize:
            print " done."

        # coefficient of regularization
        if self.visualize:
            sys.stdout.write("\r    coefficient")
            sys.stdout.flush()
        H = np.array(H)
        np_id = np.identity(min(np.array(H).shape))
        if H.shape[0] < H.shape[1]:
            Sigma = np.dot(H, H.T)
        else:
            Sigma = np.dot(H.T, H)
        if self.c is None:
            """
            print "Sigma"
            print Sigma
            print "diag"
            print np.diag(Sigma)
            coefficient = alpha * np.diag(Sigma).sum() / Sigma.shape[0]
            print "coefficient", coefficient
            regular = coefficient * np_id
            """
            regular = alpha * np.diag(np.diag(Sigma)) / Sigma.shape[0]
        elif self.c == 0:
            coefficient = 0
            regular = coefficient * np_id
        else:
            coefficient = 1. / self.c
            regular = coefficient * np_id
            
        if self.visualize:
            print " done."

        # pseudo inverse
        if self.visualize:
            sys.stdout.write("\r    pseudo inverse")
            sys.stdout.flush()
            
        Hp = np.linalg.inv(Sigma + regular)
        if H.shape[0] < H.shape[1]:
            Hp = np.dot(H.T, Hp)
        else:
            Hp = np.dot(Hp, H.T)
        if self.visualize:
            print " done."
            
        # set beta
        if self.visualize:
            sys.stdout.write("\r    set beta")
            sys.stdout.flush()
        
        self.beta = np.dot(Hp, np.array(signal))
        if self.visualize:
            print " done."

if __name__ == "__main__":
    
    train = [[1, 1, 2, 3, 4, 5, 5, 6, 5], [-8, 3, -1, -2, -4, 2, 3, -2, -2], [-3, 1, -1, 2, 3, 4, 5, 6, 7], [-2, 1, 0, 2, -3, 4, 5, -5,-2]]
    label = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    test = [[3, 3, 4, 5, 3, -2, -1, 3, 5], [-9, -3, -3, 3, 1, 4, 5, -1, 3]]

    model = ELMRegressor(n_hidden=8)#, sae_coef=[1000., 1000., 1000.])

    model.fit(train, label)

    print model.predict_batch(train)
    print model.predict(test[0])
    print model.predict_batch(test)
    print "score:", model.score(train, label)
    """
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]
    
    model = ELMClassifier()
    model.fit(train, label)
    pre = model.predict(test)
    print pre
    print model.score(train, label)
    """
