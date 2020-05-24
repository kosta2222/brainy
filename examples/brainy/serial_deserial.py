from .work_with_arr import to_ribbon
from .nn_constants import bc_bufLen, max_in_nn_1000, max_rows_orOut_10, max_stack_matrEl, max_stack_otherOp_10,\
    push_i, push_fl, make_kernel, with_bias, stop,\
    RELU, LEAKY_RELU, SIGMOID, TAN, max_spec_elems_1000,\
    determe_act_func, determe_alpha_leaky_relu, determe_alpha_sigmoid, determe_alpha_and_beta_tan, determe_in_out
import struct as st
from .NN_params import NN_params
from  .util import  get_logger
import sys
#----------------------сериализации/десериализации------------------------------
pos_bytecode=-1  # указатель на элементы байт-кода
loger=get_logger("debug", 'ser.log', __name__)
def pack_v(buffer:list, op_i, val_i_or_fl):
    """
    Добавляет в buffer буффер байт-комманды и сериализованные матричные числа как байты
    :param op_i: байт-комманда
    :param val_i_or_fl: число для серелизации - матричный элемент или количество входов выходов
    :return: следующий индекс куда можно записать команду stop
    """
    global pos_bytecode
    ops_name = ['', 'push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
                'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'determe_in_out', 'stop']  # отпечатка команд [для отладки]
    loger.debug(f"op_i {ops_name[op_i]}, {val_i_or_fl}")
    if op_i == push_fl:
        pos_bytecode += 1
        buffer[pos_bytecode] = st.pack('B', push_fl)
        for i in st.pack('<f', val_i_or_fl):
            pos_bytecode+=1
            buffer[pos_bytecode] = i.to_bytes(1, 'little')
    elif op_i == push_i:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', push_i)
        for i in st.pack('<i', val_i_or_fl):
            pos_bytecode+=1
            buffer[pos_bytecode] = i.to_bytes(1, 'little')
    elif op_i == make_kernel:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', make_kernel)
    elif op_i == with_bias:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', with_bias)
    elif op_i == determe_in_out:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', determe_in_out)
    elif op_i == determe_act_func:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', determe_act_func)
    elif op_i == determe_alpha_leaky_relu:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', determe_alpha_leaky_relu)
    elif op_i == determe_alpha_sigmoid:
        pos_bytecode+=1
        buffer[pos_bytecode] = st.pack('B', determe_alpha_sigmoid)
    elif op_i == determe_alpha_and_beta_tan:
        pos_bytecode += 1
        buffer[pos_bytecode] = st.pack('B', determe_alpha_and_beta_tan)
def to_file(nn_params:NN_params, net:list, kernel_amount, fname):
    buffer = [0] * max_spec_elems_1000  # Записываем сетевой байткод сюда подом в файл
    if pos_bytecode == len(buffer):
        print("Static memory error", end=' ')
        print("in buffer-to_file")
        sys.exit(1)

    in_=0
    out=0
    with_bias_i = 0
    stub = 0
    if nn_params.with_bias:
        with_bias_i = 1
    else:
        with_bias_i = 0
    pack_v(buffer, push_i, with_bias_i)
    pack_v(buffer, with_bias, stub)
    pack_v(buffer, push_i, nn_params.act_fu)
    pack_v(buffer, determe_act_func, stub)
    # разбираемся с параметрами активациооных функции - по умолчанию они уже заданы в nn_params
    if nn_params.act_fu == LEAKY_RELU:
        pack_v(buffer, push_fl, nn_params.alpha_leaky_relu)
        pack_v(buffer, determe_alpha_leaky_relu, stub)
    elif nn_params.act_fu == SIGMOID:
        pack_v(buffer, push_fl, nn_params.alpha_sigmoid)
        pack_v(buffer,determe_alpha_sigmoid, stub)
    elif nn_params.act_fu == TAN:
        pack_v(buffer, push_fl, nn_params.alpha_tan)
        pack_v(buffer, push_fl, nn_params.beta_tan)
        pack_v(buffer, determe_alpha_and_beta_tan, stub)

    for i in range(kernel_amount):
        in_=net[i].in_
        out=net[i].out
        pack_v(buffer, push_i,in_)
        pack_v(buffer, push_i,out)
        for row in range(out):
            for elem in range(in_):
                pack_v(buffer, push_fl,net[i].matrix[row][elem])
        pack_v(buffer, make_kernel, stub)
    dump_buffer(buffer, fname)
def dump_buffer(buffer, fname):
  global pos_bytecode
  pos_bytecode+=1
  buffer[pos_bytecode] = stop.to_bytes(1,"little")
  len_bytecode = pos_bytecode + 1
  with open(fname,'wb') as f:
           for i in range(len_bytecode):
               f.write(buffer[i])
  print("File writed")
  pos_bytecode = -1
def deserialization_vm(nn_params:NN_params, net:list, buffer:list):
     loger.debug("*in vm*")

     ops_name = ['', 'push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
                    'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'determe_in_out', 'stop']  # отпечатка команд [для отладки]
     steck_fl = [0] * 400 # стек для временного размещения элементов матриц из файла потом этот стек
        # сворачиваем в матрицу слоя после команды make_kernel
     ops_st = [0] * max_stack_otherOp_10 *2      # стек для количества входов и выходов (это целые числа)
     ip = 0
     sp_fl = -1
     sp_op = -1
     op = -1
     arg = 0
     n_lay = 0
     op = buffer[ip]
     while (op != stop):
            # print("ip",ip)
            # загружаем на стек количество входов и выходов ядра
            # чтение операции с параметром
        loger.debug(ops_name[op])
        if  op == push_i:
                v_0 = buffer[ip + 1]
                v_1 = buffer[ip + 2]
                v_2 = buffer[ip + 3]
                v_3 = buffer[ip + 4]
                arg=st.unpack('<i', bytes(list([v_0, v_1, v_2, v_3])))
                sp_op+=1
                ops_st[sp_op] = arg[0]
                ip += 4
                loger.debug(arg[0])
                # print(buffer[ip])
            # загружаем на стек элементы матриц
            # чтение операции с параметром
        elif op == push_fl:
                v_0 = buffer[ip + 1]
                v_1 = buffer[ip + 2]
                v_2 = buffer[ip + 3]
                v_3 = buffer[ip + 4]
                arg=st.unpack('<f', bytes(list([v_0, v_1, v_2, v_3])))
                sp_fl+=1
                steck_fl[sp_fl] = \
                    arg[0]
                ip += 4
                loger.debug(arg[0])
        # создаем одно ядро в массиве
        # пришла команда создать ядро
        elif op == make_kernel:
            out=ops_st[sp_op]
            sp_op-=1
            in_=ops_st[sp_op]
            sp_op-=1
            nn_params.net[n_lay].in_=in_
            nn_params.net[n_lay].out=out
            # make_kernel_f(nn_params, net, n_lay, matrix_el_st, ops_st, sp_op)
            com_el_amount=in_ * out
            for row in range(out):
                for elem in range(in_):
                    nn_params.net[n_lay].matrix[row][elem]=\
                        steck_fl[row * in_ + elem]
            sp_fl-=com_el_amount
            # переходим к следующему индексу ядра
            n_lay+=1
            # зачищаем стеки

        # пришла команда узнать пользуемся ли биасами
        # надо извлечь параметр
        elif op == with_bias:
            is_with_bias = ops_st[sp_op]
            sp_op-=1
            if is_with_bias == 1:
                nn_params.with_bias = True
            elif is_with_bias == 0:
                nn_params.with_bias = False
        elif op == determe_act_func:
            what_func = ops_st[sp_op]
            sp_op-=1
            nn_params.act_fu = what_func
        elif op == determe_alpha_and_beta_tan:
            beta = ops_st[sp_op]
            sp_op-=1
            alpha = ops_st[sp_op]
            sp_op-=1
            nn_params.alpha_tan = alpha
            nn_params.beta_tan = beta
        elif op == determe_alpha_sigmoid:
            alpha = ops_st[sp_op]
            sp_op-=1
            nn_params.alpha_sigmoid = alpha
        elif op == determe_alpha_leaky_relu:
            alpha = ops_st[sp_op]
            sp_op-=1
            nn_params.alpha_leaky_relu = alpha
        # показываем на следующую инструкцию
        ip+=1
        op = buffer[ip]

     # также подсчитаем сколько у наc ядер
     nn_params.nl_count = n_lay
     # находим количество входов
     nn_params.input_neurons = nn_params.net[0].in_ #-1  # -1 зависит от биасов
     # находим количество выходов когда образовали сеть
     nn_params.outpu_neurons=nn_params.net[nn_params.nl_count-1].out
def deserialization(nn_params:NN_params, net:list, fname:str):
    buffer = [0] * max_spec_elems_1000
    buf_str = b''
    with open(fname, 'rb') as f:
        buf_str = f.read()
        if len(buf_str)==len(buffer):
            print("Static memory error", end=' ')
            print("in buffer-deserialization")
            sys.exit(1)

    cn_by = 0
    for i in buf_str:
        buffer[cn_by] = i
        cn_by+=1
    # разборка байт-кода
    deserialization_vm(nn_params, net, buffer)
#----------------------------------------------------------------------
