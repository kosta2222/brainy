from brainy.learn import initiate_layers, cr_lay
from brainy.fit import fit
from brainy.serial_deserial import to_file, deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import SIGMOID, RELU, TAN, SOFTMAX,MODIF_MSE
from brainy.learn import answer_nn_direct, answer_nn_direct_on_contrary
import numpy as np
from brainy.util import get_logger


(push_i, push_fl, push_str, send_list, send_obj,test_of_fit_train, test_serial) = range(7)

def vm(buffer, logger=None, date=None):
    obj_net=None
    nn_params_obj=None
    len_ = 25
    if logger:
        logger.info(logger.debug(f'Log started {date}'))
    vm_is_running = True
    ip = 0
    sp = -1
    steck = [0] * len_
    op = buffer[ip]
    vm_is_running=True
    while vm_is_running:
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])  # Из строкового параметра
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])  # Из строкового параметра
        elif op == push_str:
            sp+= 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op==send_obj:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        elif op==test_of_fit_train: # 1-nn_params_obj 2-eps 3-X 4-Y 5-X_eval 6-Y_eval 7-acc_shur
            ip+=1
            arg=buffer[ip]
            nn_params_obj, eps, X, Y, X_eval, Y_eval, acc_shur=arg
            fit(nn_params_obj, eps, X, Y, X_eval, Y_eval, acc_shur, logger)
            obj_net=nn_params_obj.net
            nn_params_obj=nn_params_obj
        elif op==test_serial:
            filen=steck[sp]
            sp-=1
            to_file(nn_params_obj, obj_net, filen, logger)
        ip += 1
        if ip>(len(buffer)-1):
            return
        try:
           op = buffer[ip]
        except IndexError:
            raise RuntimeError('Maybe arg of bytecode skipped')


import unittest

class TestAnn(unittest.TestCase):
    def test_fit(self):
        loger, date = get_logger("debug", __name__, "w")
        nn_params = NN_params()
        nn_params_new = NN_params()
        i = cr_lay(nn_params, 'D', 2, 3, RELU, loger)
        i = cr_lay(nn_params, 'D', 3, 1, TAN, loger)
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.input_neurons = 2
        nn_params.outpu_neurons = 1
        # nn_params.act_fu = RELU
        nn_params.alpha_sigmoid = 0.56
        nn_params.mse_treshold = 0.0001
        nn_params.loss_func = MODIF_MSE
        X = [[1, 0], [0, 1], [1, 1], [0, 0]]
        Y = [[0], [0], [1], [0]]
        X_np = np.array(X, dtype='float32')
        Y_np = np.array(Y, dtype='float32')
        exit_code=-1
        exit_code=fit(nn_params, 10, X, Y, X, Y, 100, loger)
        self.assertEqual(0, exit_code)
        # test_of_fit_train  1-nn_params_obj 2-eps 3-X 4-Y 5-X_eval 6-Y_eval 7-acc_shur


def vm_test():
    loger, date=get_logger("debug",__name__,"w")
    nn_params = NN_params()
    nn_params_new = NN_params()
    i = cr_lay(nn_params, 'D', 2, 3, RELU,loger)
    i = cr_lay(nn_params, 'D', 3, 1, TAN,loger)
    nn_params.with_bias = False
    nn_params.with_adap_lr = False
    nn_params.lr = 0.01
    nn_params.input_neurons = 2
    nn_params.outpu_neurons = 1
    # nn_params.act_fu = RELU
    nn_params.alpha_sigmoid = 0.56
    nn_params.mse_treshold = 0.017
    nn_params.loss_func = MODIF_MSE
    X = [[1, 0], [0, 1], [1, 1], [0, 0]]
    Y = [[0], [0], [1], [0]]
    X_np = np.array(X, dtype='float32')
    Y_np = np.array(Y, dtype='float32')
    # test_of_fit_train  1-nn_params_obj 2-eps 3-X 4-Y 5-X_eval 6-Y_eval 7-acc_shur
    p1 = (test_of_fit_train,(nn_params, 10, X, Y, X, Y, 75))
    p2=(push_str,'ser_net.my',test_serial)
    vm(p1 + p2, loger, date)

if __name__ == '__main__':
   unittest.main()
   # vm_test()

