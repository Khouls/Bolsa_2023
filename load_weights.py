from definitions import *
import numpy as np

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def load_weights(model, path):
    weight_reader = WeightReader(path)
    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i))
        conv_layer.trainable = True
        
        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))
            norm_layer.trainable = True
            
            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])       
            
        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

    layer   = model.layers[-2] # last convolutional layer
    layer.trainable = True

    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
    new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

    layer.set_weights([new_kernel, new_bias])