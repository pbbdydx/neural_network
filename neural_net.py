

#improting libraries
import numpy as np 


def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation_function(layer):
    """
    input: layer - vector of values created after forward propogation
    output: 'new' layer - vector of values subject to an activation function
    """
    return sigmoid(layer) # in theory this /should/ return a numpy object if we only pass numpy objects through it


class Neuron: 
    def __init__(self, num_weights): # num_weights reflects num of outputs
        self.bias = np.random.randn()
        self.weights = [np.random.random() for i in range(num_weights)]

class Layer: 
    
    def __init__(self):
        self.input = [Neuron.weight for _ in range(3)] #creates a list of neurons that act like a layer 
        return self.input
    
    
    def forward(self, in_layer, out_layer):
        self.bias = np.rand.randint(-5, 5)
        self.input = __init__()
        self.weights = np.array([np.randon.rand(len(out_layer)) for neuron in self.input]) #create a matrix and convert it into a numpy object 
        
        if self.weights.shape[1]==self.input.shape[0]: # number of columns in the matrix matches the number of rows in the input layer
            out_layer = np.dot(self.weights, in_layer)       
        
        #apply activation function to the output layer and then add the bias 
        out_layer = activation_function(out_layer) + self.bias 
        
        return out_layer 
    