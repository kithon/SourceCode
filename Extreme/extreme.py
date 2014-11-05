# coding: utf-8
# online learning
                                  
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class ExtremeLearningMachine(object):
    def __init__(self, activation=sigmoid,
                 n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def construct(self, input, teacher):
        # construct Layer
        self.input = input
        self.teacher = teacher
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)
        self.layer = Layer(self.activation,
                           self.np_rng,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.domain)

        
    def fit(self, input, teacher):
        # construct layer
        self.construct(input, teacher)

        # convert teacher to signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])

        # fitting
        self.layer.fit(input, signal)
        
    def predict(self, input):
        # return output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(self.layer.get_output(i).tolist())
        #print "outputs", output

        predict_classes = []
        for o in predict_output:
            predict_classes.append(self.classes[o.index(max(o))])
        #print "predict" predict_classes

        return predict_classes

    def score(self, input, teacher):
        # return score
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length

class Layer(object):
    def __init__(self, activation, np_rng, size, domain):
        # initialize 
        self.activation = activation
        self.np_rng = np_rng
        self.n_input, self.n_hidden, self.n_output = size
        self.low, self.high = domain

        # initialize weight and bias
        self.w = np.array(self.np_rng.uniform(low = self.low,
                                              high = self.high,
                                              size = (self.n_input,
                                                      self.n_hidden)))
        self.b = np.array(self.np_rng.uniform(low = self.low,
                                              high = self.high,
                                              size = self.n_hidden))
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])

    def get_i2h(self, input):
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        #print "input", input
        hidden = self.get_i2h(input)
        #print "hidden", hidden
        output = self.get_h2o(hidden)
        #print "output", output
        return output
    
    def fit(self, input, signal):
        # fitting
        H = []
        for i in input:
            H.append(self.get_i2h(i))
        #print "H", H
        H = np.matrix(H)
        Hp = H.I
        Hp = np.array(Hp)

        self.beta = np.dot(Hp, np.array(signal))

        """
        # print parameter
        
        print "input", input
        print "w", self.w.T
        print "b", self.b
        print "beta", self.beta
        for i in input:
            print self.get_output(i)

        """
            

if __name__ == "__main__":
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]

    model = ExtremeLearningMachine()
    model.fit(train, label)
    pre = model.predict(test)
    print pre
    print model.score(train, label)
