import torch as pt
import torch.nn as nn

from collections import OrderedDict

# Class for Dense Neural Networks used for branch and trunk networks.
class DenseNN(nn.Module):
    def __init__(self, layers=[], dtype=pt.float32):
        super(DenseNN, self).__init__()
        print('dtype', dtype)
        
        # Create all feed-forward layers
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, pt.nn.Linear(layers[i], layers[i+1], dtype=dtype)))
            layer_list.append(('activation_%d' % i, self.activation))
        layer_list.append(('layer_%d' % (self.depth-1), pt.nn.Linear(layers[-2], layers[-1], dtype=dtype)))
        layerDict = OrderedDict(layer_list)

        # Combine all layers in a single Sequential object to keep track of parameter count
        self.layers = pt.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)
    
# Class for general DeepONets
class DeepONet(nn.Module):
    def __init__(self, eps_branch_layers, branch_layers, trunk_layers, dtype=pt.float32):
        super(DeepONet, self).__init__()

        self.branch_input_size = branch_layers[0]
        self.p = branch_layers[-1] // 2
        assert trunk_layers[-1] == 2 * self.p

        self.eps_branch_net = DenseNN(eps_branch_layers, dtype=dtype)
        self.branch_net = DenseNN(branch_layers, dtype=dtype)
        self.trunk_net = DenseNN(trunk_layers, dtype=dtype)
        
        self.params = []
        self.params.extend(self.eps_branch_net.parameters())
        self.params.extend(self.branch_net.parameters())
        self.params.extend(self.trunk_net.parameters())
        print('Number of DeepONet Parameters:', self.getNumberOfParameters())

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters())

    # The input data x = array([branch_eps, branch_x, trunk_x]) with shape (batch_size, self.branch_input_size)
    def forward(self, x):
        branch_eps_x = x[:, 0:1]
        branch_x = x[:, 1:self.branch_input_size+1]
        trunk_x = x[:, self.branch_input_size+1:]

        eps_branch_output = self.eps_branch_net.forward(branch_eps_x)
        branch_output = self.branch_net.forward(branch_x)
        trunk_output = self.trunk_net.forward(trunk_x)
     
        # Multiply element-wise and sum over axis=1 (p axis)
        u_output = pt.sum(eps_branch_output[:,:self.p] * branch_output[:,:self.p] * trunk_output[:,:self.p], dim=1)
        v_output = pt.sum(eps_branch_output[:,self.p:] * branch_output[:,self.p:] * trunk_output[:,self.p:], dim=1)
        return pt.hstack((u_output[:,None], v_output[:,None]))