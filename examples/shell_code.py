# -*-coding:utf-8-*-
import sys
from brainy.NN_params import Nn_params
from brainy.serial_deserial import deserialization
from brainy.nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN, SOFTMAX, MODIF_MSE, CROS_ENTROPY, TRESHOLD_FUNC
from brainy.serial_deserial import to_file
from brainy.fit import fit
from brainy.learn import answer_nn_direct, cr_lay, feed_forwarding_back
from brainy.util import make_train_matr, make_2d_arr, calc_out_nn, get_logger, matr_img
import numpy as np
from PIL import Image
import logging
import numpy as np
len_ = 10
stop = 0
push_i = 1
push_fl = 2
push_str = 3
push_obj = 4
fit_ = 5
predict = 6
load = 7
fit_net = 10
get_mult_class_matr = 11
determe_X_Y = 12
make_net = 13
nparray = 14
determe_X_eval_Y_eval = 15


def create_nn_params():
    return Nn_params()


nn_params = None
nn_params_new = None


def exec_(buffer, loger, date):
    X_t = None
    Y_t = None
    X_eval = None
    Y_eval = None
    ip = 0
    sp = -1
    steck = [0]*len_
    op = 0
    op = buffer[ip]
    loger.info(f'Log started at {date}')
    while True:
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])
        elif op == push_str:
            sp += 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op == push_obj:
            sp += 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op == make_net:
            loger.debug('-in make net-')
            nn_params = Nn_params()

            acts_di = {'s': SIGMOID, 'r': RELU,
                       't': TAN, 'SO': SOFTMAX, 'l': LEAKY_RELU, 'tr': TRESHOLD_FUNC}
            ip += 1
            arg = buffer[ip]
            type_m, denses, inps, acts, use_bi = arg
            if type_m == 'S':
                pass
            for i in range(len(denses)):
                if denses[i] == 'D':
                    splt_bi = use_bi[i].split('_')
                    if splt_bi[-1] == '1':
                        use_bias_ = True
                    elif splt_bi[-1] == '0':
                        use_bias_ = False
                    nn_params.input_neurons = inps[0]
                    nn_params.outpu_neurons = inps[-1]
                    nn_params = cr_lay(nn_params, 'D',
                                       inps[i], inps[i + 1],
                                       acts_di.get(acts[i]), use_bias_, loger)

        elif op == determe_X_Y:
            Y_t = steck[sp]
            sp -= 1
            X_t = steck[sp]
            sp -= 1
        elif op == determe_X_eval_Y_eval:
            Y_eval = steck[sp]
            sp -= 1
            X_eval = steck[sp]
            sp -= 1
        elif op == nparray:
            st_arg = steck[sp]
            sp -= 1
            sp += 1
            steck[sp] = np.array(st_arg)
        elif op == fit_net:
            loger.debug('-in fit net-')
            loger.debug(f'nn_params.net[0]{nn_params.net[0]}')
            loss_func_di = {"mse": MODIF_MSE, "crossentropy": CROS_ENTROPY}
            ip += 1
            arg = buffer[ip]
            eps, l_r_, with_adap_lr, ac_, mse_, loss_f, with_loss_threshold, loger = arg
            nn_params.alpha_sigmoid = 1.7159

            nn_params.loss_func = loss_func_di.get(loss_f)
            if not X_eval and not Y_eval:
                X_eval = X_t
                Y_eval = Y_t
            fit(nn_params, X_t, Y_t, X_eval, Y_eval, eps, l_r_,
                with_adap_lr, with_loss_threshold, ac_, mse_, loger)

            print("back", feed_forwarding_back(nn_params, [1, 1], loger))
        elif op == get_mult_class_matr:
            pix_am = steck[sp]
            sp -= 1
            path_ = steck[sp]
            sp -= 1
            X_t, Y_t = matr_img(path_, pix_am)
            X_t = np.array(X_t)
            Y_t = np.array(Y_t)
            X_t.astype('float32')
            Y_t.astype('float32')
            X_t /= 255
        elif op == predict:
            deserialization(nn_params_new, "to_file.my", loger)
            out_nn = answer_nn_direct(nn_params_new, X_t, loger)
            print("answer", out_nn)
        elif op == stop:
            return
        else:
            print("Unknown bytecode -> %d" % op)
            return
        ip += 1
        op = buffer[ip]


if __name__ == '__main__':
    loger = None
    if len(sys.argv) == 2:
        level = sys.argv[1]
        # level='-release'
        print("level", level)
        if level == '-debug':
            loger, date = get_logger("debug", 'log_cons.log', __name__)
        elif level == '-release':
            loger, date = get_logger("release", 'log_cons.log', __name__)
        else:
            print("Unrecognized option ", level)
            sys.exit(1)
        # np.random.seed(42)
        p1 = (push_str, r'B:\msys64\home\msys_u\code\python\brainy\examples\train_ann', push_i, 784, get_mult_class_matr,
              push_i, 15, push_fl, 100, fit_net,
              push_str, r'B:\msys64\home\msys_u\code\python\brainy\examples\ask_ann', push_i, 784, get_mult_class_matr,
              predict,
              stop)

        # eps, l_r_, with_adap_lr, ac_, mse_, loss_f, wi_loss_threshold, loger=arg fit_net
        X = [[0, 1], [1, 0], [0, 0], [1, 1]]
        Y_and = [[0], [0], [0], [1]]
        Y_or = [[1], [1], [0], [1]]
        Y_xor = [[1], [1], [0], [0]]
        X_long = [[0, 1], [1, 0], [0, 0], [1, 1],
                  [0, 1], [1, 0], [0, 0], [1, 1]]
        Y_xor_long = [[1], [1], [0], [0], [1], [1], [0], [0]]
        Y_and_so = [[0, 0], [0, 0], [0, 0], [1, 0]]

        and_p = (push_obj, X, push_obj, Y_and, determe_X_Y, push_obj, X, push_obj, Y_and, determe_X_eval_Y_eval,
                 make_net, ('S', ('D', 'D'), (2, 2, 1), ('tr', 'tr'),
                            ('usebias_0', 'usebias_0')),
                 fit_net, (100, 0.1, False, 100, 0, "mse", True, loger),
                 stop)
        and_p_long = (push_obj, X, push_obj, Y_and, determe_X_Y, push_obj, X, push_obj, Y_and, determe_X_eval_Y_eval,
                      make_net, ('S', ('D', 'D', 'D'), (2, 4, 4, 1), ('tr', 'tr', 'tr'),
                                 ('usebias_0', 'usebias_0', 'usebias_0')),
                      fit_net, (100, 0.1, False, 100, 0, "mse", True, loger),
                      stop)
        xor_p = (push_obj, X_long, push_obj, Y_xor_long, determe_X_Y, push_obj, X_long, push_obj, Y_xor_long, determe_X_eval_Y_eval,
                 make_net, ('S', ('D', 'D'), (2, 8, 1), ('t', 't'),
                            ('usebias_0', 'usebias_0')),
                 fit_net, (100, 0.1, False, 100, 0.001, "mse", True, loger),
                 stop)

        and_p_sm = (push_obj, X, push_obj, Y_and_so, determe_X_Y, push_obj, X, push_obj, Y_and_so, determe_X_eval_Y_eval,
                    make_net, ('S', ('D', 'D'), (2, 2, 2), ('tr', 'SO'),
                               ('usebias_0', 'usebias_0')),
                    fit_net, (100, 0.1, False, 100, 0, "crossentropy", True, loger),
                    stop)

        exec_(and_p_sm, loger, date)
    else:
        print("Program must have option: -release or -debug")
        sys.exit(-2)
