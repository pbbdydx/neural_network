#improting libraries
import numpy as np 

def calc_error(y_true, y_pred): ### Produces the root means squared error
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
    ### Adding each layer to layers
    layers.append([Neuron(vector[1]) for num in range(first_layer_count)])
    neuron_index = 2
    for layer_count in middle_layers: 
        layers.append([Neuron(vector[neuron_index]) for num in range(layer_count)])
        neuron_index += 1
    layers.append([Neuron(vector[-1]) for num in range(last_layer_count)]) ### Placeholder
    ### (We will need a separate type of output neuron that has no weights)
    return layers 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation_function(layer):
    """
    input: layer - vector of values created after forward propogation
    output: 'new' layer - vector of values subject to an activation function
    """
    return sigmoid(layer) ### in theory this /should/ return a numpy object if we only pass numpy objects through it


class Neuron: 
    def __init__(self, num_weights): ### num_weights reflects num of outputs
        self.bias = np.random.randn()
        self.weights = [np.random.random() for i in range(num_weights)]

class Layer: 
    
    def __init__(self, in_list): ### Creates a list of neurons that act like a layer 
        self.input = in_list 
        
    def __iter__(self): ### Allows the list to be iterated over which helps with the next step
        return iter(self.input)
    
    def forward(self,next_layer):
        ### Create a matrix for dot product
        matrix_of_weights = []
        
        print(self.input)
        for neuron in self:
            matrix_of_weights.append(neuron.weights)
        
        ### turn matrix of weights and the next layer into a numpy array for easy computation 
        # matrix_of_weights = np.array(matrix_of_weights)
        # print(matrix_of_weights)
        # next_layer = np.array(next_layer)
        
        ### Chatgpt suggestion ----- will try some more testing with this tomorrow 
        
        # for neuron in self: 
        #     print(neuron.weights)
        #     matrix_of_weights.append(neuron.weights)
        
        # turn matrix of weights and the next layer into a numpy array for easy computation 
        matrix_of_weights = np.array(matrix_of_weights)
        print("Matrix of weights:")
        print(matrix_of_weights)
        
        next_layer_inputs = np.array([neuron.weights for neuron in next_layer]).T
        print("Next layer inputs (transposed):")
        print(next_layer_inputs)
        
        ### Doing the forward propogating
        if matrix_of_weights.shape[1]==next_layer_inputs.shape[0]: ### Number of columns in the matrix must match the number of rows in the input layer
            out_layer = np.dot(matrix_of_weights, in_layer)
        else: raise ValueError(f"The dimensions of the matrix and next layer do not allow for a compatible matrix multiplication .\nShape of matrix: {matrix_of_weights.shape}\nShape of next layer: {next_layer_inputs.shape}. ")
        
        ### Apply activation function to the output layer and then add the bias 
        out_layer = activation_function(out_layer) + self.bias ### Have to change this too. instead of adding a constant add the vector of biases
        
        
        return out_layer 

#testing 
if __name__=="__main__":
    net = create_layers([3,4,2,3])
    # for layer in net:
    #     print(layer)
    first_layer = Layer(net[0])
    second_layer = Layer(net[1])
    print(first_layer.forward(second_layer))
    
    
    