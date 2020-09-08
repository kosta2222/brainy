from .nn_constants import RELU_DERIV, RELU, TRESHOLD_FUNC, TRESHOLD_FUNC_DERIV, LEAKY_RELU, LEAKY_RELU_DERIV,\
    SIGMOID, SIGMOID_DERIV, DEBUG, DEBUG_STR, INIT_W_HE, INIT_W_GLOROT_MY, INIT_W_HABR, INIT_W_MY, INIT_W_UNIFORM,\
    TAN, TAN_DERIV, INIT_W_HE_MY, INIT_W_MY_DEB, INIT_W_HE, INIT_RANDN
from .NN_params import Nn_params
from .nn_constants import max_rows_orOut_10
import math
import numpy as np
import random


def softmax_ret_vec(x: list, rows):
    out_vec = [0]*max_rows_orOut_10
    sum_exp = 0
    for i in range(rows):
        sum_exp += math.exp(x[i])
    for i in range(rows):
        out_vec[i] = math.exp(x[i]) / sum_exp
    return out_vec


ready = False
y = 0
# операции для функций активаций и их производных
np.random.randn()


def operations(op, a, b, c, d, str_, nn_params: Nn_params):
    global ready, y
    alpha_leaky_relu = nn_params.alpha_leaky_relu
    alpha_sigmoid = nn_params.alpha_sigmoid
    alpha_tan = nn_params.alpha_tan
    beta_tan = nn_params.beta_tan
    if op == RELU:
        if (a <= 0):
            return 0
        else:
            return a
    elif op == RELU_DERIV:
        if (a <= 0):
            return 0
        else:
            return 1
    elif op == TRESHOLD_FUNC:
        if (a > 0.5):
            return 1
        else:
            return 0
    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (a <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == LEAKY_RELU_DERIV:
        if (a <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == SIGMOID:
        y = 1 / (1 + math.exp(- alpha_sigmoid * a))
        return y
    elif op == SIGMOID_DERIV:

        return (alpha_sigmoid * y * (1 - y))
    elif op == INIT_W_HE_MY:
        if ready:
            ready = False
            return -math.sqrt(2 / a) * 0.567141530112327
        ready = True
        return math.sqrt(2 / a) * 0.567141530112327
    elif op == INIT_W_MY:
        return random.random()
    elif op == TAN:
        f = alpha_tan * math.tanh(beta_tan * a)
        return f
    elif op == TAN_DERIV:
        return beta_tan / alpha_tan * (alpha_tan * alpha_tan - y * y)
    elif op == INIT_W_MY_DEB:
        return 2
    elif op == INIT_RANDN:
        return np.random.randn()
    else:
        print("Op or function does not support ", op)
