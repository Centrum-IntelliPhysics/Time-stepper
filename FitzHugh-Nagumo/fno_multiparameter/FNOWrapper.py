import torch as pt

import main

import __main__
setattr(__main__, "FNO1d", main.FNO1d)
setattr(__main__, "SpectralConv1d", main.SpectralConv1d)

class FNOWrapper:
    def __init__(self):
        self.modes = 100
        self.width = 32

        self.model_dir = './Results/model/'
        self.network = pt.load(self.model_dir + 'FHN_BF_multiparameter')
        print(type(self.network))

    def __call__(self, u, v, eps):
        # Preprocessing x = (u, v, eps) stacked in three dimensions
        x = pt.transpose(pt.vstack((u, v, eps * pt.ones_like(u))), 0, 1)[None,:,:]

        # FNO propagation
        output = self.network.forward(x)

        # Postprocessing x = (u, v, eps)
        print('output shape', output.shape)
        return output[0,:,0], output[0,:,1]