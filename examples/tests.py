#-*-coding: cp1251-*-
import unittest
from unittest import TestCase
from brainy.learn import initiate_layers
from brainy.fit import fit
from brainy.serial_deserial import to_file, deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import SIGMOID, RELU, TAN
from brainy.learn import answer_nn_direct, answer_nn_direct_on_contrary
import numpy as np
from brainy.util import get_logger
import pdb
# pdb.set_trace()
logger=get_logger("debug","log.txt",__name__)
# Просто работа скрипта
class Tests_my(TestCase):
    def setUp(self) -> None:
        pass
        # self.nn_params=NN_params()
        # self.nn_params_new=NN_params()
        # self.buffer_ser=[0]*256*2

    def test_and(self):

        nn_params=NN_params()
        nn_params_new=NN_params()
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.act_fu = RELU
        nn_params.alpha_sigmoid = 0.056
        nn_params.mse_treshold = 0.0017
        nn_map = (2,3,1)
        initiate_layers(nn_params, nn_map, len(nn_map))
        X=[[1,0],[0,1],[1,1],[0,0]]
        Y=[[0],[0],[1],[0]]

        X_np=np.array(X, dtype='float64')
        Y_np=np.array(Y, dtype='float64')
        # X_np-=np.mean(X_np, axis=0, dtype='float64')
        # Y_np-=np.mean(Y_np, axis=0, dtype='float64')
        X_np=np.std(X_np, axis=0)
        Y_np=np.std(Y_np, axis=0)
        fit(nn_params,10,X,Y,X,Y,100,logger)
        to_file(nn_params,nn_params.net, logger, 'wei_and.my')
        deserialization(nn_params_new, nn_params_new.net, 'wei_and.my')
        out_nn_dir = answer_nn_direct(nn_params_new, [1, 1], 1)
        self.assertGreater(out_nn_dir[0], 0.5)
        out_nn_dir = answer_nn_direct(nn_params, [1, 1], 1)
        self.assertGreater(out_nn_dir[0], 0.5)
        out_nn_contr_ser = answer_nn_direct_on_contrary(nn_params_new, [1], 1)
        out_nn_contr = answer_nn_direct_on_contrary(nn_params, [1], 1)
        self.assertAlmostEqual(out_nn_contr_ser[0],out_nn_contr[0])
        self.assertAlmostEqual(out_nn_contr_ser[1],out_nn_contr[1],places=4)


    def test_or(self):
        nn_params=NN_params()
        nn_params_new=NN_params()
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.act_fu = RELU
        nn_params.alpha_sigmoid = 0.056
        nn_params.mse_treshold = 0.0017
        nn_map = (2, 3, 1)
        initiate_layers(nn_params, nn_map, len(nn_map))
        X = [[1, 0], [0, 1], [1, 1], [0, 0]]
        Y = [[1], [1], [1], [0]]

        X_np = np.array(X, dtype='float64')
        Y_np = np.array(Y, dtype='float64')
        # X_np-=np.mean(X_np, axis=0, dtype='float64')
        # Y_np-=np.mean(Y_np, axis=0, dtype='float64')
        X_np = np.std(X_np, axis=0)
        Y_np = np.std(Y_np, axis=0)
        fit(nn_params, 10, X, Y, X, Y, 100,logger)
        to_file(nn_params, nn_params.net, logger, 'wei_or.my')
        deserialization(nn_params_new, nn_params_new.net, 'wei_or.my')
        out_nn_dir = answer_nn_direct(nn_params_new, [1, 1], 1)
        self.assertGreater(out_nn_dir[0], 0.5)
        out_nn_dir = answer_nn_direct(nn_params, [1, 1], 1)
        self.assertGreater(out_nn_dir[0], 0.5)
        out_nn_contr_ser = answer_nn_direct_on_contrary(nn_params_new, [1], 1)
        out_nn_contr = answer_nn_direct_on_contrary(nn_params, [1], 1)
        self.assertAlmostEqual(out_nn_contr_ser[0],out_nn_contr[0])
        self.assertAlmostEqual(out_nn_contr_ser[1],out_nn_contr[1],places=4)

if __name__ == '__main__':
    unittest.main()
