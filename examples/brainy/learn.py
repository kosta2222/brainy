import math
from .nn_constants import RELU, RELU_DERIV, INIT_W_HE, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV, INIT_W_GLOROT_MY,\
INIT_W_HE_MY, SOFTMAX
from .NN_params import NN_params   # импортруем параметры сети
from .Lay import Lay   # импортируем слой
from .work_with_arr import copy_vector
from .operations import operations, softmax_ret_vec
def calc_out_error(nn_params:NN_params,objLay:Lay, targets:list):
    for row in range(objLay.out):
        nn_params.out_errors[row] = (objLay.hidden[row] - targets[row]) * operations(nn_params.act_fu + 1, objLay.cost_signals[row], 0.42, 0, 0, "", nn_params)
def calc_hid_error(nn_params:NN_params, objLay:Lay, essential_gradients:list, entered_vals:list):
  try:
    for elem in range(objLay.in_):
        for row in range(objLay.out):
            objLay.errors[elem]+=\
                essential_gradients[row] * \
                objLay.matrix[row][elem]  * \
                operations(nn_params.act_fu + 1, entered_vals[elem], 0, 0, 0, "", nn_params)
  except Exception as e:
      print("in calc hid err Exc")
      print("el",elem)
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
def get_cost_signals(objLay:Lay):
    return objLay.cost_signals
def get_hidden(objLay:Lay):
    return objLay.hidden
def get_essential_gradients(objLay:Lay):
    return objLay.errors
def calc_hid_zero_lay(zeroLay:Lay,essential_gradients:list):
    for elem in range(zeroLay.in_):
        for row in range(zeroLay.out):
            zeroLay.errors[elem]+=\
                essential_gradients[row] * zeroLay.matrix[row][elem]
def upd_matrix(nn_params:NN_params, objLay:Lay, entered_vals):
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            objLay.matrix[row][elem]-= nn_params.lr * objLay.errors[elem] * entered_vals[elem]
def feed_forwarding(nn_params:NN_params,ok:bool, debug):
    make_hidden(nn_params, nn_params.net[0], nn_params.inputs, debug)
    for i in range(1,nn_params.nl_count):
        make_hidden(nn_params, nn_params.net[i], get_hidden(nn_params.net[i - 1]), debug)
    if ok:
        for i in range(nn_params.outpu_neurons):
            pass
        return nn_params.net[nn_params.nl_count-1].hidden
    else:
         backpropagate(nn_params)
def feed_forwarding_on_contrary(nn_params:NN_params, ok:bool, debug):
    make_hidden_on_contrary(nn_params, nn_params.net[nn_params.nl_count - 1 ], nn_params.inputs, debug)
    for i in range(nn_params.nl_count - 2, -1, -1):
        make_hidden_on_contrary(nn_params, nn_params.net[i], get_hidden(nn_params.net[i + 1]), debug)
    if ok:
        for i in range(nn_params.input_neurons):
            pass
            # print("%d item val %f"%(i + 1,nn_params.net[0].hidden[i]))
        return nn_params.net[0].hidden
def train(nn_params:NN_params,in_:list,targ:list, debug):
    copy_vector(in_,nn_params.inputs,nn_params.input_neurons)
    copy_vector(targ,nn_params.targets,nn_params.outpu_neurons)
    feed_forwarding(nn_params,False, debug)
def answer_nn_direct(nn_params:NN_params,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.input_neurons)
    out_nn=feed_forwarding(nn_params,True, debug)
    return out_nn
def answer_nn_direct_on_contrary(nn_params:NN_params,in_:list, debug):
    out_nn = None
    copy_vector(in_,nn_params.inputs,nn_params.outpu_neurons)
    out_nn=feed_forwarding_on_contrary(nn_params,True, debug)
    return out_nn
# Получить вектор входов, сделать матричный продукт и матричный продукт пропустить через функцию активации,
# записать этот вектор в параметр слоя сети(hidden)
def make_hidden(nn_params, objLay:Lay, inputs:list, debug):
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
           val = operations(nn_params.act_fu,tmp_v, 0, 0, 0, "", nn_params)
           objLay.hidden[row] = val
        tmp_v = 0
    if objLay.act_func==SOFTMAX:
        objLay.hidden=softmax_ret_vec(objLay.cost_signals,Lay.out)
def make_hidden_on_contrary(nn_params:NN_params, objLay:Lay, inputs:list, debug):
    tmp_v = 0
    val = 0
    for row in range(objLay.out):
        for elem in range(objLay.in_):
            if nn_params.with_bias:
               if elem == 0:
                  tmp_v+=objLay.matrix[row][elem]
               else:
                  tmp_v+=objLay.matrix[row][elem]\
                         * inputs[elem]
            else:
                tmp_v+=objLay.matrix[row][elem] * inputs[elem]
        objLay.cost_signals[elem] = tmp_v
        val = operations(nn_params.act_fu, tmp_v, 0, 0, 0, "", nn_params)
        objLay.hidden[elem] = val
        tmp_v = 0
def backpropagate(nn_params:NN_params):
    calc_out_error(nn_params, nn_params.net[nn_params.nl_count - 1],nn_params.targets)
    for i in range(nn_params.nl_count - 1, 0, -1):
        if i == nn_params.nl_count - 1:
           calc_hid_error(nn_params, nn_params.net[i], nn_params.out_errors, get_cost_signals(nn_params.net[i - 1]))
        else:
            calc_hid_error(nn_params, nn_params.net[i], get_essential_gradients(nn_params.net[i + 1]), get_cost_signals(nn_params.net[i - 1]))
    calc_hid_zero_lay(nn_params.net[0], get_essential_gradients(nn_params.net[1]))
    for i in range(nn_params.nl_count - 1, 0, -1):
        upd_matrix(nn_params, nn_params.net[i],  get_cost_signals(nn_params.net[i - 1]))
    upd_matrix(nn_params, nn_params.net[0], nn_params.inputs)
# заполнить матрицу весов рандомными значениями по He, исходя из количесва входов и выходов,
# записать результат в вектор слоев(параметр matrix), здесь проблема матрица неправильно заполняется
def set_io(nn_params:NN_params, objLay:Lay, inputs, outputs):
    objLay.in_=inputs
    objLay.out=outputs
    for row in range(outputs):
        for elem in range(inputs):
            objLay.matrix[row][elem] = operations(INIT_W_MY, inputs+1, outputs, 0, 0, "", nn_params)
def initiate_layers(nn_params:NN_params,network_map:tuple,size):
    in_ = 0
    out = 0
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
        set_io(nn_params, nn_params.net[i], in_, out)
