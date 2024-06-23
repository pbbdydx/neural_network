#improting libraries
import numpy as np 

def calc_error(y_true, y_pred): # Produces the root means squared error
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be equal")
    total = 0
    for i in range(len(y_true)):
        total += (1/len(y_true))*(((y_true[i] - y_pred[i])**2))
    return total**0.5

def create_layers(vector: list[int]):
    '''
    the vector stores the number of neurons in each layer
    function returns a list of lists of neurons 
    '''
    
    first_layer_count = vector[0]
    last_layer_count = vector[-1]
    middle_layers= vector[1:-1] 
    layers=[]
    # Adding each layer to layers
    layers.append([Neuron() for num in range(first_layer_count)])
    for layer_count in middle_layers: 
        layers.append([Neuron() for num in range(layer_count)])
    layers.append([Neuron() for num in range(last_layer_count)]) 
    return layers 

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

#testing 
if __name__=="__main__":
    net = create_layers([3,4,2,3])
    for layer in net:
        print(layer)
    
    
    