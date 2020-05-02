import math
import numpy as np
import struct as st
from .nn_constants import max_in_nn, max_trainSet_rows, max_validSet_rows, max_rows_orOut, max_am_layer\
, max_am_epoch, max_am_objMse, max_stack_matrEl, max_stack_otherOp, bc_bufLen
from .nn_constants import RELU, RELU_DERIV, INIT_W_HE, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, INIT_W_GLOROT_MY,\
INIT_W_HE_MY
from .NN_params import NnParams   # импортруем параметры сети
from .Nn_lay import nnLay   # импортируем слой
from .work_with_arr import copy_vector
from .operations_func import operations
from .util_func import _0_
# import pdb
# pdb.set_trace()
def calc_out_error(nn_params:NnParams,objLay:nnLay, targets:list):
    """
    Вычислить градиентную ошибку на выходном слое,записать этот параметр-вектор в обьект nnLay выходного слоя
    в параметр errors
    :param objLay: обьект слоя
    :param targets: вектор-ответы от учителя
    :return:
    """
    for row in range(objLay.out):
        nn_params.out_errors[row] = (objLay.hidden[row] - targets[row]) * operations(nn_params.act_fu + 1, objLay.cost_signals[row], 0.42, 0, 0, "", nn_params)
def calc_hid_error(nn_params:NnParams, objLay:nnLay, essential_gradients:list, entered_vals:list):
    for elem in range(objLay.in_):
        for row in range(objLay.out):
            objLay.errors[elem]+=essential_gradients[row] * objLay.matrix[row][elem]  * operations(nn_params.act_fu + 1, entered_vals[elem], 0, 0, 0, "", nn_params)
    # print("in calc_hid_error essential_gradients",essential_gradients)
    # print("in calc_hid_error entered_vals",entered_vals)
    # print("in calc_hid_error errors",objLay.errors)
    # print("in calc_hid_error matrix",objLay.matrix)
def get_min_square_err(out_nn:list,teacher_answ:list,n):
    sum=0
    for row in range(n):
        sum+=math.pow((out_nn[row] - teacher_answ[row]),2)
    return sum / n
def get_mean(l1:list, l2:list, n):
    sum=0
    for row in range(n):
        sum+=l1[row]- l2[row]
    return sum / n
def get_cost_signals(objLay:nnLay):
    return objLay.cost_signals
def get_hidden(objLay:nnLay):
    return objLay.hidden
def get_essential_gradients(objLay:nnLay):
    return objLay.errors
def calc_hid_zero_lay(zeroLay:nnLay,essential_gradients:list):
    for elem in range(zeroLay.in_):
        for row in range(zeroLay.out):
            zeroLay.errors[elem]+=essential_gradients[row] * zeroLay.matrix[row][elem]
    # print("in calc_hid_zero_lay",zeroLay.errors)
def upd_matrix(nn_params:NnParams, objLay:nnLay, entered_vals):
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            objLay.matrix[row][elem]-= nn_params.lr * objLay.errors[elem] * entered_vals[elem]
    # print("in upd_matrix matrix state",objLay.matrix)
def feed_forwarding(nn_params:NnParams,ok:bool, debug:int):
    make_hidden(nn_params, nn_params.list_[0], nn_params.inputs, debug)
    for i in range(1,nn_params.nlCount):
        make_hidden(nn_params, nn_params.list_[i], get_hidden(nn_params.list_[i - 1]), debug)
    if ok:
        for i in range(nn_params.outputNeurons):
            print("%d item val %f"%(i + 1,nn_params.list_[nn_params.nlCount - 1].hidden[i]))
        return nn_params.list_[nn_params.nlCount - 1].hidden
    else:
         backpropagate(nn_params)
def feed_forwarding_on_contrary(nn_params:NnParams, ok:bool, debug:int):
    make_hidden_on_contrary(nn_params, nn_params.list_[nn_params.nlCount - 1 ], nn_params.inputs, debug)
    # print("in feed_forwarding_on_contrary nn_params.nlCount - 1.out",nn_params.list_[nn_params.nlCount - 1 ].out)
    for i in range(nn_params.nlCount - 2, -1, -1):
        make_hidden_on_contrary(nn_params, nn_params.list_[i], get_hidden(nn_params.list_[i + 1]), debug)
    if ok:
        for i in range(nn_params.inputNeurons):
            print("%d item val %f"%(i + 1,nn_params.list_[0].hidden[i]))
        return nn_params.list_[0].hidden
# для теста, создать один слой
def create_one_lay():
    lay=nnLay()
    lay.in_=3
    lay.out=2
    lay.matrix=[[-1,1,4],[3,4,-7]]
    return lay
def train(nn_params:NnParams,in_:list,targ:list, debug):
    copy_vector(in_,nn_params.inputs,nn_params.inputNeurons)
    copy_vector(targ,nn_params.targets,nn_params.outputNeurons)
    # print("in train in_ vec",in_)
    # print("in train targ vec",targ)
    feed_forwarding(nn_params,False, debug)
def answer_nn_direct(nn_params:NnParams,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.inputNeurons)
    # print("in answer_nn in_ vec",in_)
    out_nn=feed_forwarding(nn_params,True, debug)
    return out_nn
def answer_nn_direct_on_contrary(nn_params:NnParams,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.outputNeurons)
    # print("in answer_nn in_ vec",in_)
    out_nn=feed_forwarding_on_contrary(nn_params,True, debug)
    return out_nn
# Получить вектор входов, сделать матричный продукт и матричный продукт пропустить через функцию активации,
# записать этот вектор в параметр слоя сети(hidden)
def make_hidden(nn_params, objLay:nnLay, inputs:list, debug):
    # print("in make_hidden inputs",inputs)
    tmp_v = 0
    val = 0
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            if nn_params.with_bias:
               if elem==1:
                  tmp_v+=objLay.matrix[row][elem]
               else:
                  tmp_v+=objLay.matrix[row][elem] *\
                         inputs[elem]
            else:
                tmp_v+=objLay.matrix[row][elem] *\
                       inputs[elem]
        objLay.cost_signals[row] = tmp_v
        val = operations(nn_params.act_fu,tmp_v, 0, 0, 0, "", nn_params)
        objLay.hidden[row] = val
        tmp_v = 0
    # print("in make_hidden e",objLay.cost_signals)
    # print("in make_hidden h",objLay.hidden)
    # print("in make_hidden matrix state",objLay.matrix)
def make_hidden_on_contrary(nn_params:NnParams, objLay:nnLay, inputs:list, debug):
    # print("in make_hidden_on_contrary ")
    # print("inputs",inputs)
    tmp_v = 0
    val = 0
    # print("in_",objLay.in_,"\nout",objLay.out)
    # print("matrix",objLay.matrix)
    for elem in range(objLay.in_):
        # print("elem",elem)
        for row in range(objLay.out):
            if nn_params.with_bias:
               if elem == 1: 
                  tmp_v+=objLay.matrix[row][elem] * inputs[row]
               else:
                  tmp_v+=objLay.matrix[row][elem] * inputs[elem]
            else:
                tmp_v+=objLay.matrix[row][elem] * inputs[row]
            # print("tmp_v",tmp_v)
        objLay.cost_signals[elem] = tmp_v
        val = operations(nn_params.act_fu, tmp_v, 0, 0, 0, "", nn_params)
        objLay.hidden[elem] = val
        tmp_v = 0
        val = 0
    # print("in make_hidden e",objLay.cost_signals)
    # print("in make_hidden h",objLay.hidden)
def backpropagate(nn_params:NnParams):
    calc_out_error(nn_params, nn_params.list_[nn_params.nlCount - 1],nn_params.targets)
    for i in range(nn_params.nlCount - 1, 0, -1):
        if i == nn_params.nlCount - 1:
           calc_hid_error(nn_params, nn_params.list_[i], nn_params.out_errors, get_cost_signals(nn_params.list_[i - 1]))
        else:
            calc_hid_error(nn_params, nn_params.list_[i], get_essential_gradients(nn_params.list_[i + 1]), get_cost_signals(nn_params.list_[i - 1]))
    calc_hid_zero_lay(nn_params.list_[0], get_essential_gradients(nn_params.list_[1]))
    for i in range(nn_params.nlCount - 1, 0, -1):
        upd_matrix(nn_params, nn_params.list_[i],  get_cost_signals(nn_params.list_[i - 1]))
    upd_matrix(nn_params, nn_params.list_[0], nn_params.inputs)
# заполнить матрицу весов рандомными значениями по He, исходя из количесва входов и выходов,
# записать результат в вектор слоев(параметр matrix), здесь проблема матрица неправильно заполняется
def set_io(nn_params:NnParams, objLay:nnLay, inputs, outputs):
    objLay.in_=inputs
    objLay.out=outputs
    for row in range(outputs):
        for elem in range(inputs):
            objLay.matrix[row][elem] = operations(INIT_W_MY, inputs+1, outputs, 0, 0, "", nn_params)
    # print("in set_io matrix", objLay.matrix)
def initiate_layers(nn_params:NnParams,network_map:tuple,size):
    """
    инициализировать вектор слоев используя функцию set_io исходя из кортежа, который должен описывать
    например количество входов, сколько нейронов(элементов) в следующем слое, сколько выводов
    :param network_map: кортеж карта слоев
    :param size: размер кортежа
    :return: None
    """
    in_ = 0
    out = 0
    nn_params.nlCount = size - 1
    nn_params.inputNeurons = network_map[0]
    nn_params.outputNeurons = network_map[nn_params.nlCount]
    set_io(nn_params, nn_params.list_[0],network_map[0],network_map[1])
    for i in range(1, nn_params.nlCount ):# след. матр. д.б. (3,1) т.е. in(elems)=3 out(rows)=1
        if nn_params.with_bias:
           in_ = network_map[i] + 1
        else:
            in_ = network_map[i]
        out = network_map[i + 1]
        set_io(nn_params, nn_params.list_[i], in_, out)
