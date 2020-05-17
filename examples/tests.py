from unittest import TestCase
from brainy.learn import initiate_layers
from brainy.fit import fit
from brainy.serial_deserial import to_file, deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import SIGMOID, RELU, TAN

class Tests(TestCase):
    def setUp(self) -> None:
        self.nn_params=NN_params()
        self.buffer_ser=[0]*256*2

    def test_and(self):
        self.nn_params.with_bias = False
        self.nn_params.with_adap_lr = True
        self.nn_params.lr = 0.01
        self.nn_params.act_fu = TAN
        self.nn_params.alpha_sigmoid = 0.056
        self.nn_params.mse_treshold = 0.0
        nn_map = (2,3,1)
        initiate_layers(self.nn_params, nn_map, len(nn_map))
        X=[[1,0],[0,1],[1,1],[0,0]]
        Y=[[0],[0],[1],[0]]
        fit(self.buffer_ser,self.nn_params,10,X,Y,X,Y,75,use_logger='debug')

