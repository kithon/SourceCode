# coding: utf-8
# online learning
                                  
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def sign(x):
    return (np.sign(x - 0.5) + 1) / 2

class MLELMClassifier(object):
    """
    Multi-Layer Extreme Learning Machine

    
    """
    
    def __init__(self, activation=sigmoid, n_hidden=[]):
        print "__init__"

    def pre_train(self, input):
        print "pre_train"

        
    def get_extraction(self, input):
        print "get_extractation"
        self.n_input = len(input[0])

        
    def fit(self, input, teacher):
        print "fit"
        

class ELMClassifier(object):
    """
    Extreme Learning Machine
    
    
    """

    def __init__(self, activation=sigmoid, vector='orthogonal', regular=True,
                 c=0., n_hidden=50, seed=123, domain=[-1., 1.]):
        # initialize
        self.activation = activation
        self.vector = vector
        self.regular = regular
        self.c = c
        self.n_hidden = n_hidden
        self.np_rng = np.random.RandomState(seed)
        self.domain = domain
        
    def construct(self, input, teacher):
        # set parameter of layer
        self.input = input
        self.teacher = teacher
        classes = []
        for t in teacher:
            if not t in classes:
                classes.append(t)
        self.classes = classes
        self.n_input = len(input[0])
        self.n_output = len(self.classes)
        low, high = self.domain

        # set weight and bias (randomly)
        weight = self.np_rng.uniform(low = low, high = high, size = (self.n_input, self.n_hidden))
        bias = self.np_rng.uniform(low = low, high = high, size = self.n_hidden)

        if self.vector == 'orthogonal':
            # orthogonal weight and forcely regularization
            print "set weight and bias orthogonaly"
            for i in xrange(len(weight)):
                w = weight[i]
                for j in xrange(0,i):
                    w = w - weight[j].dot(w) * weight[j]
                w = w / np.linalg.norm(w)
                weight[i] = w

            if self.regular:
                # bias regularization
                denom = np.linalg.norm(bias)
                if denom != 0:
                    denom = bias / denom
            
            
        elif self.vector == 'random':
            # randomly and regulatization
            print "set weight and bias randomly"
            if self.regular:
                #for i,w enumerate(weight.T):
                for i,w in enumerate(weight):
                    denom = np.linalg.norm(w)
                    if denom != 0:
                        #weight.T[i] = w / denom
                        weight[i] = w / denom

                # bias regularization
                denom = np.linalg.norm(bias)
                if denom != 0:
                    bias = bias / denom
            
        else:
            print "warning: vector isn't orthogonal or random"
            
        
        # set weight and bias
        self.weight = weight
        self.bias = bias     
        
            
        # initialize layer
        self.layer = Layer(self.activation,
                           [self.n_input, self.n_hidden, self.n_output],
                           self.weight,
                           self.bias,
                           self.c)

        
    def fit(self, input, teacher):
        # construct layer
        self.construct(input, teacher)

        # convert teacher to signal
        signal = []
        id_matrix = np.identity(self.n_output).tolist()
        for t in teacher:
            signal.append(id_matrix[self.classes.index(t)])

        # fit layer
        self.layer.fit(input, signal)
        
    def predict(self, input):
        # get predict_output
        predict_output = []
        for i in input:
            o = self.layer.get_output(i).tolist()
            predict_output.append(self.layer.get_output(i).tolist())
        #print "outputs", predict_output

        # get predict_classes from index of max_function(predict_output) 
        predict_classes = []
        for o in predict_output:
            predict_classes.append(self.classes[o.index(max(o))])
        #print "predict" predict_classes

        return predict_classes

    def score(self, input, teacher):
        # get score 
        count = 0
        length = len(teacher)
        predict_classes = self.predict(input)
        for i in xrange(length):
            if predict_classes[i] == teacher[i]: count += 1
        return count * 1.0 / length

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_beta(self):
        return self.layer.beta

class Layer(object):
    def __init__(self, activation, size, w, b, c):
        # initialize 
        self.activation = activation
        self.n_input, self.n_hidden, self.n_output = size
        self.c = c
        self.w = w
        self.b = b
        self.beta = np.zeros([self.n_hidden,
                              self.n_output])

    def get_beta(self):
        return self.beta
        
    def get_i2h(self, input):
        # activation from input to hidden
        return self.activation(np.dot(self.w.T, input) + self.b)

    def get_h2o(self, hidden):
        # activation from hidden to output
        return np.dot(self.beta.T, hidden)

    def get_output(self, input):
        # activation from input to output
        hidden = self.get_i2h(input)
        output = self.get_h2o(hidden)
        return output
    
    def fit(self, input, signal):
        # set self.beta from activation
        H = []
        for i in input:
            H.append(self.get_i2h(i))
        H = np.matrix(H)
        if self.c == 0:
            # Tend to be Memory Error
            Hp = H.T * (H * H.T).I
            #Hp = H.I
        else:
            id_matrix = np.matrix(np.identity(len(input)))
            Hp = H.T * ((id_matrix / (self.c * 1.)) + H * H.T).I            
        Hp = np.array(Hp)
        self.beta = np.dot(Hp, np.array(signal))


if __name__ == "__main__":
    train = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    label = [1, 1, -1, -1]
    test = [[3, 3], [-3, -3]]

    model = ELMClassifier()
    model.fit(train, label)
    pre = model.predict(test)
    print pre
    print model.score(train, label)
