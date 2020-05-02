from .work_with_arr import copy_matrixAsStaticSquare_toRibon
from .nn_constants import bc_bufLen, max_in_nn, max_rows_orOut, max_stack_matrEl, max_stack_otherOp,\
    push_i, push_fl, make_kernel, with_bias, stop,\
    RELU, LEAKY_RELU, SIGMOID, TAN,\
    determe_act_func, determe_alpha_leaky_relu, determe_alpha_sigmoid, determe_alpha_and_beta_tan
from .Nn_lay import nnLay
import struct as st
from .NN_params import NN_params
from .util_func import _0_
#----------------------сериализации/десериализации------------------------------
pos_bytecode=0  # указатель на элементы байт-кода 
def compil_serializ(nn_params:NN_params, b_c:list, list_:nnLay, kernel_amount, f_name):
    print("in compil_serializ")
    in_=0
    out=0
    with_bias_i = 0
    stub = 0
    matrix=[0]*(max_in_nn * max_rows_orOut)
    if nn_params.with_bias:
        with_bias_i = 1
    else:
        with_bias_i = 0
    py_pack(b_c, push_i, with_bias_i)
    py_pack(b_c, with_bias, stub)
    py_pack(b_c, push_i, nn_params.act_fu)
    py_pack(b_c, determe_act_func, stub)
    # разбираемся с параметрами активациооных функции - по умолчанию они уже заданы в nn_params
    if nn_params.act_fu == LEAKY_RELU:
        py_pack(b_c, push_fl, nn_params.alpha_leaky_relu)
        py_pack(b_c, determe_alpha_leaky_relu, stub)
    elif nn_params.act_fu == SIGMOID:
        py_pack(b_c, push_fl, nn_params.alpha_sigmoid)
        py_pack(b_c,determe_alpha_sigmoid, stub)
    elif nn_params.act_fu == TAN:
        py_pack(b_c, push_fl, nn_params.alpha_tan)
        py_pack(b_c, push_fl, nn_params.beta_tan)
        py_pack(b_c, determe_alpha_and_beta_tan, stub)
    print("list_",nn_params.list_)

    for i in range(kernel_amount):
        print("*list_ in",list_[i].in_)
        in_=list_[i].in_
        out=list_[i].out
        py_pack(b_c, push_i,in_)
        py_pack(b_c, push_i,out)
        copy_matrixAsStaticSquare_toRibon(list_[i].matrix, matrix, in_, out)
        matrix_elems = in_ * out


        for j in range(matrix_elems):
            py_pack(b_c, push_fl, matrix[j])
        py_pack(b_c, make_kernel, stub)


    dump_bc(b_c, f_name)
def py_pack (b_c:list, op_i, val_i_or_fl):
    """
    Добавляет в b_c буффер байт-комманды и сериализованные матричные числа как байты
    :param op_i: байт-комманда
    :param val_i_or_fl: число для серелизации - матричный элемент или количество входов выходов
    :return: следующий индекс куда можно записать команду stop
    """
    global pos_bytecode
    ops_name = ['push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
    'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'stop']  # отпечатка команд [для отладки]
    print("in py_pack op",ops_name[op_i],"val_i_or_fl",val_i_or_fl)
    print("pos b_c",pos_bytecode)
    try:
        if op_i == push_fl:
          # try:
            b_c[pos_bytecode] = st.pack('B', push_fl)
            pos_bytecode+=1
            for i in st.pack('<f', val_i_or_fl):
                b_c[pos_bytecode] = i.to_bytes(1, 'little')
                pos_bytecode+=1
          # except Exception:
          #     print("pos b_c",pos_bytecode)
        elif op_i == push_i:
            b_c[pos_bytecode] = st.pack('B', push_i)
            pos_bytecode+=1
            b_c[pos_bytecode] = st.pack('B', val_i_or_fl)
            pos_bytecode+=1
        elif op_i == make_kernel:
            b_c[pos_bytecode] = st.pack('B', make_kernel)
            pos_bytecode+=1
        elif op_i == with_bias:
            b_c[pos_bytecode] = st.pack('B', with_bias)
            pos_bytecode+=1
        elif op_i == with_bias:
            b_c[pos_bytecode] = st.pack('B', with_bias)
            pos_bytecode+=1
        elif op_i == determe_act_func:
            b_c[pos_bytecode] = st.pack('B', determe_act_func)
            pos_bytecode+=1
        elif op_i == determe_alpha_leaky_relu:
            b_c[pos_bytecode] = st.pack('B', determe_alpha_leaky_relu)
            pos_bytecode+=1
        elif op_i == determe_alpha_sigmoid:
            b_c[pos_bytecode] = st.pack('B', determe_alpha_sigmoid)
            pos_bytecode+=1
        elif op_i == determe_alpha_and_beta_tan:
            b_c[pos_bytecode] = st.pack('B', determe_alpha_and_beta_tan)
        pos_bytecode+=1
    except Exception:
        print("Except")
        print("b_c pos",pos_bytecode)
def  dump_bc(b_c, f_name):
  global pos_bytecode
  b_c[pos_bytecode] = stop.to_bytes(1,"little")
  pos_bytecode+=1
  # try:
  with open(f_name,'wb') as f:
       print("b_c",b_c)
       len_bytecode = pos_bytecode
       for i in range(len_bytecode):
           print("i", b_c[i])
           f.write(b_c[i])
  pos_bytecode = 0
  # except Exception:
  #     print("i",i)
  #     return
def make_kernel_f(nn_params:NN_params, list_:list, lay_pos, matrix_el_st:list,  ops_st:list,  sp_op):
    """
    Создает  ядро в векторе слоев
    :param list_: ссылка на вектор слоев
    :param lay_pos: позиция слоя (int)
    :param matrix_el_st: ссылка на стек матричных элементов
    :param ops_st: ссылка на стек входов/выходов
    :param sp_op: вершина стека входов/выходов(int)
    :return:
    """
    out = ops_st[sp_op]
    in_ = ops_st[sp_op - 1]
    list_[lay_pos].out = out
    list_[lay_pos].in_ = in_
    for  row in range(out):
        for elem in range(in_):
            list_[lay_pos].matrix[row][elem] = matrix_el_st[row * elem]   # десериализированная матрица
    _0_("make_kernel")
def vm_to_deserialize(nn_params:NN_params, list_:list, bin_buf:list):
    """
    Элемент виртуальной машины чтобы в вектор list_ матриц весов
    записать десериализированные из файла матрицы весов и смочь
    пользоваться этим вектором для предсказания.
    :param list_: вектор матриц весов
    :param bin_buf: список байт - комманд из файла
    :return:
    """
    print("in vm_to_deserialize")
    ops_name = ['push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
                'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'stop']  # отпечатка команд [для отладки]
    matrix_el_st = [0] * max_stack_matrEl # стек для временного размещения элементов матриц из файла потом этот стек
    # сворачиваем в матрицу слоя после команды make_kernel
    ops_st = [0] * max_stack_otherOp      # стек для количества входов и выходов (это целые числа)
    ip = 0
    sp_ma = -1
    sp_op = -1
    op = -1
    arg = 0
    n_lay = 0
    op = bin_buf[ip]
    while (op != stop):
        # загружаем на стек количество входов и выходов ядра
        # чтение операции с параметром
        print(ops_name[op],end=' ')
        if  op == push_i:
            sp_op+=1
            ip+=1
            print("arg",bin_buf[ip])
            ops_st[sp_op] = bin_buf[ip]
        # загружаем на стек элементы матриц
        # чтение операции с параметром
        elif op == push_fl:
            i_0 = bin_buf[ip + 1]
            i_1 = bin_buf[ip + 2]
            i_2 = bin_buf[ip + 3]
            i_3 = bin_buf[ip + 4]
            arg=st.unpack('<f', bytes(list([i_0, i_1, i_2, i_3])))
            sp_ma+=1
            matrix_el_st[sp_ma] = arg[0]
            ip += 4
        # создаем одно ядро в массиве
        # пришла команда создать ядро
        elif op == make_kernel:
            make_kernel_f(nn_params, list_, n_lay, matrix_el_st, ops_st, sp_op)
            # переходим к следующему индексу ядра
            n_lay+=1
            # зачищаем стеки
            sp_op = -1
            sp_ma = -1
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
        op = bin_buf[ip]
    # также подсчитаем сколько у наc ядер
    nn_params.nlCount = n_lay
    # находим количество входов
    nn_params.inputNeurons = nn_params.list_[0].in_ #-1  # -1 зависит от биасов
    # находим количество выходов когда образовали сеть
    nn_params.outputNeurons=nn_params.list_[nn_params.nlCount-1].out
    _0_("vm")
def deserializ(nn_params:NN_params, list_:list, f_name:str):
    bin_buf = [0] * bc_bufLen
    buf_str = b''
    with open(f_name, 'rb') as f:
        buf_str = f.read()
    j = 0
    for i in buf_str:
        bin_buf[j] = i
        j+=1
    # разборка байт-кода
    vm_to_deserialize(nn_params, list_, bin_buf)
    _0_("vm_deserializ")
#----------------------------------------------------------------------
