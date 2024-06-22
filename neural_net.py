

#improting libraries
import numpy as np 

class Layer: 
    
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self,in_layer, out_layer):
        self.bias = np.rand.randn(-5, 5)
        self.input = in_layer
        self.weights = np.array([np.randon.rand(len(out_layer)) for neuron in self.input]) #create a matrix and convert it into a numpy object 
        
        if self.weights.shape[1]==self.input.shape[0]: # number of columns in the matrix matches the number of rows in the input layer
            out_layer = np.dot(self.weights, in_layer)       
        
        
        return out_layer