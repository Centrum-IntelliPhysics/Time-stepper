import torch as pt
import torch.nn as nn

from collections import OrderedDict

# Class for Dense Neural Networks used for branch and trunk networks.
class DenseNN(nn.Module):
    def __init__(self, layers=[]):
        super(DenseNN, self).__init__()
        
        # Create all feed-forward layers
        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, pt.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), pt.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # Combine all layers in a single Sequential object to keep track of parameter count
        self.layers = pt.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)
    
# Class for general DeepONets
class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()

        self.branch_input_size = 401
        self.p = branch_layers[-1] // 2
        assert trunk_layers[-1] == 2 * self.p

        self.branch_net = DenseNN(branch_layers)
        self.trunk_net = DenseNN(trunk_layers)
        
        self.params = []
        self.params.extend(self.branch_net.parameters())
        self.params.extend(self.trunk_net.parameters())
        print('Number of DeepONet Parameters:', sum(p.numel() for p in self.parameters()))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters())

    # The input data x = array([branch_x, trunk_x]) with shape (batch_size, 402)
    def forward(self, x):
        branch_x = x[:, 0:self.branch_input_size]
        trunk_x = x[:, self.branch_input_size:]
        branch_output = self.branch_net.forward(branch_x)
        trunk_output = self.trunk_net.forward(trunk_x)
     
        # Multiply element-wise and sum over axis=1 (p axis)
        u_output = pt.sum(branch_output[:,0:self.p] * trunk_output[:,0:self.p], dim=1)
        v_output = pt.sum(branch_output[:,self.p:] * trunk_output[:,self.p:], dim=1)
        return pt.hstack((u_output[:,None], v_output[:,None]))