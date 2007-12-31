import math
from .nn_constants import RELU, RELU_DERIV, INIT_W_HE, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, INIT_W_GLOROT_MY,\
INIT_W_HE_MY, SOFTMAX, CROS_ENTROPY, MODIF_MSE, INIT_W_MY_DEB
from .NN_params import NN_params   # импортруем параметры сети
from .Lay import Lay, Dense   # импортируем слой
from .operations import operations, softmax_ret_vec
from .work_with_arr import copy_vector
import logging


def calc_out_error(nn_params:NN_params,objLay:Lay, targets:list, loger:logging.Logger)->list:
    out_errors=None
    if objLay.act_func!=SOFTMAX and nn_params.loss_func==MODIF_MSE:
      for row in range(objLay.out):
        nn_params.out_errors[row] = (objLay.hidden[row] - targets[row]) * operations(objLay.act_func + 1, objLay.cost_signals[row], 0.42, 0, 0, "", nn_params)
    elif objLay.act_func==SOFTMAX and nn_params.loss_func==CROS_ENTROPY:
        for row in range(objLay.out):
            nn_params.out_errors[row] = objLay.hidden[row] - targets[row]
    out_errors=nn_params.out_errors
    return out_errors



def calc_hid_error(nn_params:NN_params,prev_left_layer:Lay, current_layer_index:int, next_right_layer_deltas:list, loger:logging.Logger)->list:
    """
    Calcs deltas for current layer
    :param nn_params: Whole ann params
    :param prev_left_layer: prev_left_layer
    :param current_layer_index: current layer index that must provide right answer to righter layer or out
    :param right_deltas or list of tangens if it is out of nn:
    :param loger: loger
    :action creates right deltas for this layer
    """
    current_layer=nn_params.net[current_layer_index]
    for elem in range(current_layer.in_):
         for row in range(current_layer.out):
              current_layer.errors[elem] +=current_layer.matrix[row][elem] * \
              next_right_layer_deltas[row] * \
              operations(prev_left_layer.act_func + 1, prev_left_layer.cost_signals[elem], 0, 0, 0, "", nn_params)
    return current_layer

def get_min_square_err(out_nn:list,teacher_answ:list,n:int)->float:
    sum=0
    for row in range(n):
        sum+=math.pow((out_nn[row] - teacher_answ[row]),2)
    return sum / n


def get_cros_entropy(ans, targ, n)->float:
    E=0
    for row in range(n):
        if targ[row]==1:
            E-=math.log(ans[row],math.e)
        else:
            E-=(1-math.log(ans[row],math.e))
    return E


def get_mean(l1:list, l2:list, n)->float:
    sum=0
    for row in range(n):
        sum+=l1[row]- l2[row]
    return sum / n


def get_cost_signals(objLay:Lay)->list:
    return objLay.cost_signals


def get_hidden(objLay:Lay)->list:
    return objLay.hidden


def get_essential_gradients(objLay:Lay)->list:
    return objLay.errors


def calc_hid_zero_lay(zeroLay:Lay,past_right_lay:Lay)->list:
    errors=None
    for elem in range(zeroLay.in_):
        for row in range(zeroLay.out):
            zeroLay.errors[elem]+=past_right_lay.errors[row] * zeroLay.matrix[row][elem]
    errors=zeroLay.errors
    return errors


def upd_matrix(nn_params:NN_params, objLay:Lay, entered_vals)->list:
    matrix=None
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            if nn_params.with_bias:
                if elem==0:
                   objLay.matrix[row][elem]-= nn_params.lr * objLay.errors[elem] * 1
                else:
                    objLay.matrix[row][elem]-= nn_params.lr * objLay.errors[elem] * entered_vals[row]
            else:
                objLay.matrix[row][elem] -= nn_params.lr * objLay.errors[elem] * entered_vals[row]
    matrix=objLay.matrix
    return matrix

def feed_forwarding(nn_params:NN_params,ok:bool, loger)->int:
    if nn_params.nl_count==1:
       make_hidden(nn_params, nn_params.net[0], nn_params.inputs, loger)
    else:
      make_hidden(nn_params, nn_params.net[0], nn_params.inputs, loger)
      for i in range(1,nn_params.nl_count):
        make_hidden(nn_params, nn_params.net[i], get_hidden(nn_params.net[i - 1]), loger)
    if ok:
        for i in range(nn_params.outpu_neurons):
            pass
        return nn_params.net[nn_params.nl_count-1].hidden
    else:
         backpropagate(nn_params, loger)
    return 0


def feed_forwarding_on_contrary(nn_params:NN_params, ok:bool, loger:logging.Logger)->list:
    hidden=None
    make_hidden_on_contrary(nn_params, nn_params.net[nn_params.nl_count - 1 ], nn_params.inputs, loger)
    for i in range(nn_params.nl_count - 2, -1, -1):
        make_hidden_on_contrary(nn_params, nn_params.net[i], get_hidden(nn_params.net[i + 1]), loger)
    if ok:
        for i in range(nn_params.input_neurons):
            pass
            # print("%d item val %f"%(i + 1,nn_params.net[0].hidden[i]))
        return nn_params.net[0].hidden
    hidden=nn_params[0].hidden
    return hidden


def train(nn_params:NN_params,in_:list,targ:list, loger:logging.Logger)->int:
    copy_vector(in_,nn_params.inputs,nn_params.input_neurons)
    copy_vector(targ,nn_params.targets,nn_params.outpu_neurons)
    feed_forwarding(nn_params,False, loger)
    return 0


def answer_nn_direct(nn_params:NN_params,in_:list, loger:logging.Logger)->list:
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.input_neurons)
    out_nn=feed_forwarding(nn_params,True, loger)
    return out_nn


def answer_nn_direct_on_contrary(nn_params:NN_params,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.outpu_neurons)
    out_nn=feed_forwarding_on_contrary(nn_params,True, debug)
    return out_nn


# Получить вектор входов, сделать матричный продукт и матричный продукт пропустить через функцию активации,
# записать этот вектор в параметр слоя сети(hidden)
def make_hidden(nn_params, objLay:Lay, inputs:list, loger:logging.Logger):
    loger.debug('-in make_hidden-')
    loger.debug(f'lay {objLay.des}')
    loger.debug(f'use func: {objLay.act_func}')
    if objLay.des=='d':
        tmp_v = 0
        val = 0
        for row in range(objLay.out):
            for elem in range(objLay.in_):
                if nn_params.with_bias:
                   if elem==0:
                      tmp_v+=objLay.matrix[row][elem]
                   else:
                      tmp_v+=objLay.matrix[row][elem] * inputs[elem]
                else:
                    tmp_v+=objLay.matrix[row][elem] * inputs[elem]
            objLay.cost_signals[row] = tmp_v
            if objLay.act_func!=SOFTMAX:
               val = operations(objLay.act_func,tmp_v, 0, 0, 0, "", nn_params)
               objLay.hidden[row] = val
            tmp_v = 0
        if objLay.act_func==SOFTMAX:
            ret_vec=softmax_ret_vec(objLay.cost_signals,objLay.out)
            copy_vector(ret_vec, objLay.hidden, objLay.out )
        loger.debug(f'cost s : {objLay.cost_signals[:10]}')
        loger.debug(f'hid s : {objLay.hidden[:10]}')


def make_hidden_on_contrary(nn_params:NN_params, objLay:Lay, inputs:list, loger:logging.Logger):
    tmp_v = 0
    val = 0
    tmp_v=0
    if objLay.des=='d':
        tmp_v=objLay.in_
        objLay.in_=objLay.out
        objLay.out=tmp_v
        for row in range(objLay.out):
            for elem in range(objLay.in_):
                if nn_params.with_bias:
                   if elem == 0:
                      tmp_v+=objLay.matrix[row][elem]
                   else:
                      tmp_v+=objLay.matrix[row][elem] * inputs[elem]
                else:
                    tmp_v+=objLay.matrix[row][elem] * inputs[elem]
            objLay.cost_signals[row] = tmp_v
            val = operations(nn_params.act_fu, tmp_v, 0, 0, 0, "", nn_params)
            objLay.hidden[row] = val
            tmp_v = 0
            if objLay.act_func == SOFTMAX:
                   ret_vec = softmax_ret_vec(objLay.cost_signals, objLay.out)
                   copy_vector(ret_vec, objLay.hidden, objLay.out)


def backpropagate(nn_params:NN_params, loger):
    calc_out_error(nn_params, nn_params.net[nn_params.nl_count - 1],nn_params.targets, loger)
    for i in range(nn_params.nl_count - 1, 0, -1):
        if i == nn_params.nl_count - 1:
           calc_hid_error(nn_params, nn_params.net[i-1], i, nn_params.out_errors, loger)
        else:
            calc_hid_error(nn_params, nn_params.net[i-1], i, nn_params.net[i+1].errors, loger)
    calc_hid_zero_lay(nn_params.net[0], nn_params.net[1])
    for i in range(nn_params.nl_count - 1, 0, -1):
        upd_matrix(nn_params, nn_params.net[i],  get_hidden(nn_params.net[i - 1]))
    upd_matrix(nn_params, nn_params.net[0], nn_params.inputs)


# заполнить матрицу весов рандомными значениями по He, исходя из количесва входов и выходов,
# записать результат в вектор слоев(параметр matrix), здесь проблема матрица неправильно заполняется
def set_io(nn_params:NN_params, objLay:Lay, inputs, outputs)->list:
    matrix=None
    objLay.in_=inputs
    objLay.out=outputs
    for row in range(outputs):
        for elem in range(inputs):
            objLay.matrix[row][elem] = operations(INIT_W_MY, inputs+1, outputs, 0, 0, "", nn_params)
    matrix=objLay.matrix
    return matrix


def initiate_layers(nn_params:NN_params,network_map:tuple,size:int)->list:
    in_ = 0
    out = 0
    matrix=None
    nn_params.nl_count = size - 1
    nn_params.input_neurons = network_map[0]
    nn_params.outpu_neurons = network_map[nn_params.nl_count]
    set_io(nn_params, nn_params.net[0],network_map[0],network_map[1])
    for i in range(1, nn_params.nl_count ):# след. матр. д.б. (3,1) т.е. in(elems)=3 out(rows)=1
        if nn_params.with_bias:
           in_ = network_map[i] + 1
        else:
            in_ = network_map[i]
        out = network_map[i + 1]
        matrix=set_io(nn_params, nn_params.net[i], in_, out)
    return matrix


def cr_lay(nn_params:NN_params, type_='D', in_=0, out=0, act_func=None, loger=None)->int:
    i=0
    if type_=='D':
        nn_params.sp_l+=1
        i=nn_params.sp_l
        nn_params.sp_d+=1
        dense=nn_params.denses[nn_params.sp_d]
        nn_params.net[i]=dense
        nn_params.net[i].in_=in_
        nn_params.net[i].out=out
        nn_params.net[i].act_func=act_func
        set_io(nn_params,nn_params.net[i],in_,out)
        nn_params.nl_count+=1
        loger.debug(nn_params.net[i])
        return i