from unittest import TestCase
from brainy.learn import initiate_layers
from brainy.fit import fit
from brainy.serial_deserial import to_file, deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import SIGMOID, RELU, TAN
from brainy.learn import answer_nn_direct, answer_nn_direct_on_contrary

class Tests(TestCase):
    def setUp(self) -> None:
        self.nn_params=NN_params()
        self.nn_params_new=NN_params()
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
        fit(self.buffer_ser,self.nn_params,10,X,Y,X,Y,100,use_logger='debug')
        to_file(self.nn_params, self.buffer_ser,self.nn_params.net, 2, 'wei_and.my')
        deserialization(self.nn_params_new,self.nn_params_new.net,'wei_and.my')
        out_nn_dir=answer_nn_direct(self.nn_params_new, [1, 1], 1)
        print("out-nn-dir",out_nn_dir)
        out_nn_contr=answer_nn_direct_on_contrary(self.nn_params_new,[1],1)
        print("out-nn-contr",out_nn_contr)
if __name__ == '__main__':
 o=Tests()
 o.main()

