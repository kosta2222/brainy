import math
from .nn_constants import RELU, RELU_DERIV, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV,\
    SOFTMAX, CROS_ENTROPY, MODIF_MSE, INIT_RANDN
from .NN_params import Nn_params   # импортруем параметры сети
from .Lay import Lay, Dense   # импортируем слой
from .work_with_arr import copy_vector
from .operations import operations, softmax_ret_vec
from .work_with_arr import copy_vector
import logging
import numpy as np


def calc_out_error(nn_params: Nn_params, targets: list, loger: logging.Logger):
    layer = nn_params.net[nn_params.nl_count-1]
    out = layer.out
    if layer.act_func != SOFTMAX and nn_params.loss_func == MODIF_MSE:
        tmp_v = 0
        for row in range(out):
            tmp_v += (layer.hidden[row] - targets[row]) * operations(
                layer.act_func+1, layer.hidden[row], nn_params)
        nn_params.out_errors[row] = tmp_v

    elif layer.act_func == SOFTMAX and nn_params.loss_func == CROS_ENTROPY:
        tmp_v = 0
        for row in range(out):
            tmp_v += (layer.hidden[row] - targets[row])
        nn_params.out_errors[row] = tmp_v


def calc_hid_error(nn_params: Nn_params, layer_ind: int, errors_next: list, layer_hidden: list, loger: logging.Logger):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        tmp_v = 0
        for elem in range(layer.in_):
            tmp_v += layer.matrix[row][elem] * errors_next[row] * operations(
                layer.act_func + 1, layer_hidden[row], nn_params)
        layer.errors[row] = tmp_v


def upd_matrix(nn_params: Nn_params, layer_ind, errors_next, inputs, lr, loger):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        error_next = errors_next[row]
        for elem in range(layer.in_):
            if layer.with_bias:
                if elem == 0:
                    layer.matrix[row][elem] -= lr * error_next * 1
                else:
                    layer.matrix[row][elem] -= lr * \
                        error_next * inputs[elem]

            else:
                layer.matrix[row][elem] -= lr * \
                    error_next *\
                    inputs[elem]


def backpropagate(nn_params: Nn_params, out_nn, targets, inputs, lr, loger):
    calc_out_error(nn_params, targets, loger)
    j = nn_params.nl_count
    layer_last = nn_params.net[j - 1]
    for i in range(j - 1, 0, - 1):
        layer = nn_params.net[i]
        layer_next = nn_params.net[j + 1]
        if i == j - 1:
            calc_hid_error(nn_params, i, calc_diff(out_nn, targets, layer.out),
                           layer_last.hidden, loger)
        else:
            calc_hid_error(nn_params, i, layer_next.errors,
                           layer.hidden, loger)

    for i in range(j - 1, 0, - 1):
        layer_prev = nn_params.net[i - 1]
        layer_next = nn_params.net[i + 1]
        if i == j - 1:
            upd_matrix(nn_params, i, nn_params.out_errors,
                       layer_prev.hidden, lr, loger)
        else:
            upd_matrix(nn_params, i, layer_next.errors,
                       layer_prev.hidden, lr, loger)

    j = 0
    layer = nn_params.net[j + 1]
    upd_matrix(nn_params, j, layer.errors, inputs, lr, loger)


def get_min_square_err(out_nn: list, teacher_answ: list, n):
    sum = 0
    for row in range(n):
        sum += math.pow((out_nn[row] - teacher_answ[row]), 2)
    return sum / n


def calc_diff(out_nn, teacher_answ, n):
    diff = [0] * len(out_nn)
    for row in range(n):
        diff[row] = out_nn[row] - teacher_answ[row]
    return diff


def get_err(diff: list):
    sum = 0
    for row in range(len(diff)):
        sum += diff[row] * diff[row]
    return sum


def get_mse(out_nn, teacher, n):
    """
    Получить среднеквадратичную ошибку сети
    out_nn: вектор выхода сети
    teacher: вектор ответов
    n: количество элементов в любом векторе
    return ошибку
    """
    sum_ = 0
    for row in range(n):
        sum_ += math.pow((out_nn[row]-teacher[row]), 2)
    return sum_ / n


def get_cros_entropy(ans, targ, n):
    E = 0
    for row in range(n):
        if targ[row] == 1:
            E -= math.log(ans[row], math.e)
        else:
            E -= (1-math.log(ans[row], math.e))
    return E


def get_mean(l1: list, l2: list, n):
    sum = 0
    for row in range(n):
        sum += l1[row] - l2[row]
    return sum / n


def get_cost_signals(objLay: Lay):
    return objLay.cost_signals


def get_hidden(objLay: Lay):
    return objLay.hidden


def feed_forwarding(nn_params: Nn_params, inputs, loger):
    make_hidden(nn_params, 0, inputs, loger)
    j = nn_params.nl_count
    for i in range(1, j):
        inputs = get_hidden(nn_params.net[i - 1])
        make_hidden(nn_params, i, inputs, loger)

    last_layer = nn_params.net[j-1]

    return get_hidden(last_layer)


def answer_nn_direct(nn_params: Nn_params, inputs, loger):
    out_nn = None
    out_nn = feed_forwarding(nn_params, inputs, loger)
    return out_nn


def make_hidden(nn_params, layer_ind, inputs: list, loger: logging.Logger):
    layer = nn_params.net[layer_ind]
    if layer.des == 'd':
        val = 0
        for row in range(layer.out):
            tmp_v = 0
            for elem in range(layer.in_):
                if layer.with_bias:
                    if elem == 0:
                        tmp_v += layer.matrix[row][elem] * 1
                    else:
                        tmp_v += layer.matrix[row][elem] * inputs[elem]
                else:
                    tmp_v += layer.matrix[row][elem] * inputs[elem]
            layer.cost_signals[row] = tmp_v

            if layer.act_func != SOFTMAX:
                val = operations(layer.act_func, tmp_v, nn_params)
                layer.hidden[row] = val

        if layer.act_func == SOFTMAX:
            ret_vec = softmax_ret_vec(layer.cost_signals, layer.out)
            copy_vector(ret_vec, layer.hidden, layer.out)


def make_hidden_on_contrary(nn_params: Nn_params, objLay: Lay, inputs: list, loger: logging.Logger):
    tmp_v = 0
    val = 0
    tmp_v = 0
    if objLay.des == 'd':
        tmp_v = objLay.in_
        objLay.in_ = objLay.out
        objLay.out = tmp_v
        for row in range(objLay.out):
            for elem in range(objLay.in_):
                if objLay.wi_bias:
                    if elem == 0:
                        tmp_v += objLay.matrix[row][elem]
                    else:
                        tmp_v += objLay.matrix[row][elem] * inputs[elem]
                else:
                    tmp_v += objLay.matrix[row][elem] * inputs[elem]
            objLay.cost_signals[row] = tmp_v
            val = operations(nn_params.act_fu, tmp_v, nn_params)
            objLay.hidden[row] = val
            tmp_v = 0
            if objLay.act_func == SOFTMAX:
                ret_vec = softmax_ret_vec(objLay.cost_signals, objLay.out)
                copy_vector(ret_vec, objLay.hidden, objLay.out)


def cr_lay(nn_params: Nn_params, type_='D', in_=0, out=0, act_func=None, with_bias=True, loger=None):
    if type_ == 'D':
        nn_params.sp_d += 1
        layer = nn_params.net[nn_params.sp_d]
        layer.in_ = in_
        layer.out = out
        layer.act_func = act_func

        if with_bias:
            in_ += 1
            layer.with_bias = True

        for row in range(out):
            for elem in range(in_):
                layer.matrix[row][elem] = operations(
                    INIT_W_MY, 0, nn_params)

        nn_params.nl_count += 1
        return nn_params
