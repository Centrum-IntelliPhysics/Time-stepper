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

    def __call__(self, x):
        return self.network.forward(x)