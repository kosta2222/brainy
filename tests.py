from unittest import TestCase
from .learn import initiate_layers
from .fit import fit
from .serial_deserial import to_file, deserialization
from .NN_params import NN_params
from .nn_constants import SIGMOID

class Tests(TestCase):
    def setUp(self) -> None:
        self.nn_params=NN_params()
        self.buffer_ser=[0]*256*2

    def test_and(self):
        self.nn_params.with_bias = True
        self.nn_params.with_adap_lr = True
        self.nn_params.lr = 0.01
        self.nn_params.act_fu = SIGMOID
        self.nn_params.alpha_sigmoid = 0.056
        self.mse_treshold = 0.001
        nn_map = (2,8,1)
        initiate_layers(self.nn_params, nn_map, len(nn_map))
        X=[[1,0],[0,1],[1,1],[0,0]]
        Y=[[0],[0],[1],[0]]
        fit(self.buffer_ser,self.nn_params,X,Y,X,Y,100)

