from .work_with_arr import to_ribbon
from .nn_constants import bc_bufLen, max_in_nn_1000, max_rows_orOut_10, max_stack_matrEl, max_stack_otherOp_10,\
    push_i, push_fl, make_kernel, with_bias, stop,\
    RELU, LEAKY_RELU, SIGMOID, TAN, max_spec_elems_1000,\
    determe_act_func, determe_alpha_leaky_relu, determe_alpha_sigmoid, determe_alpha_and_beta_tan, determe_in_out
import struct as st
# ----------------------сериализации/десериализации------------------------------
pos_bytecode = -1  # указатель на элементы байт-кода
loger = None


def pack_v(buffer, op_i, val_i_or_fl, loger):
    """
    Добавляет в buffer буффер байт-комманды и сериализованные матричные числа как байты
    :param op_i: байт-комманда
    :param val_i_or_fl: число для серелизации - матричный элемент или количество входов выходов
    :return: следующий индекс куда можно записать команду stop
    """
    try:
        global pos_bytecode
        ops_name = ['', 'push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
                    'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'determe_in_out', 'stop']  # отпечатка команд [для отладки]
        loger.debug(f"op_i {ops_name[op_i]}, {val_i_or_fl}")
        if op_i == push_fl:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', push_fl)
            for i in st.pack('<f', val_i_or_fl):
                pos_bytecode += 1
                buffer[pos_bytecode] = i.to_bytes(1, 'little')
        elif op_i == push_i:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', push_i)
            for i in st.pack('<i', val_i_or_fl):
                pos_bytecode += 1
                buffer[pos_bytecode] = i.to_bytes(1, 'little')
        elif op_i == make_kernel:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', make_kernel)
        elif op_i == with_bias:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', with_bias)
        elif op_i == determe_in_out:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', determe_in_out)
        elif op_i == determe_act_func:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', determe_act_func)
        elif op_i == determe_alpha_leaky_relu:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', determe_alpha_leaky_relu)
        elif op_i == determe_alpha_sigmoid:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', determe_alpha_sigmoid)
        elif op_i == determe_alpha_and_beta_tan:
            pos_bytecode += 1
            buffer[pos_bytecode] = st.pack('B', determe_alpha_and_beta_tan)
    except IndexError:
        print("Static memory error:", end=' ')
        print("in buffer (where we put bytecode to serelialize net)")
        print("[init in serial_deserial.tofile()]")
        loger.debug(
            "Static memory error: in buffer (where we put bytecode to serelialize net)[init in serial_deserial.tofile()])")
        return


def to_file(nn_params, net, logger, fname):
    # Записываем сетевой байткод сюда потом в файл
    buffer = [0] * max_spec_elems_1000 * 1000
    in_ = 0
    out = 0
    with_bias_i = 0
    stub = 0
    # pack_v(buffer, push_i, with_bias_i, logger)
    # pack_v(buffer, with_bias, stub, logger)
    # разбираемся с параметрами активациооных функции - по умолчанию они уже заданы в nn_params
    pack_v(buffer, push_fl, nn_params.alpha_leaky_relu, logger)
    pack_v(buffer, determe_alpha_leaky_relu, stub, logger)
    pack_v(buffer, push_fl, nn_params.alpha_sigmoid, logger)
    pack_v(buffer, determe_alpha_sigmoid, stub, logger)
    pack_v(buffer, push_fl, nn_params.alpha_tan, logger)
    pack_v(buffer, push_fl, nn_params.beta_tan, logger)
    pack_v(buffer, determe_alpha_and_beta_tan, stub, logger)
    
    lenn = nn_params.nl_count
    for i in range(lenn):
        layer = net[i]
        pack_v(buffer, push_i, nn_params.net[i].act_func, logger)
        if layer.with_bias:    
            with_bias_i = 1 
        else:
            with_bias_i = 0
        pack_v(buffer, push_i, with_bias_i, logger)  
        in_ = layer.in_
        out = layer.out
        pack_v(buffer, push_i, in_, logger)
        pack_v(buffer, push_i, out, logger)
        for row in range(out):
            for elem in range(in_):
                pack_v(buffer, push_fl, layer.matrix[row][elem], logger)
        pack_v(buffer, make_kernel, stub, logger)
    dump_buffer(buffer, fname)


def dump_buffer(buffer, fname):
    global pos_bytecode
    pos_bytecode += 1
    buffer[pos_bytecode] = stop.to_bytes(1, "little")
    len_bytecode = pos_bytecode + 1
    with open(fname, 'wb') as f:
        for i in range(len_bytecode):
            f.write(buffer[i])
    pos_bytecode = -1


def deserialization_vm(nn_params, buffer: list, loger):
    loger.debug("- in vm -")

    ops_name = ['', 'push_i', 'push_fl', 'make_kernel', 'with_bias', 'determe_act_func', 'determe_alpha_leaky_relu',
                'determe_alpha_sigmoid', 'determe_alpha_and_beta_tan', 'determe_in_out', 'stop']  # отпечатка команд [для отладки]
    # стек для временного размещения элементов матриц из файла потом этот стек
    steck_fl = [0] * max_spec_elems_1000
    # сворачиваем в матрицу слоя после команды make_kernel
    # стек для количества входов и выходов (это целые числа)
    ops_st = [0] * max_stack_otherOp_10 * 2
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

        if op == push_i:
            v_0 = buffer[ip + 1]
            v_1 = buffer[ip + 2]
            v_2 = buffer[ip + 3]
            v_3 = buffer[ip + 4]
            arg = st.unpack('<i', bytes(list([v_0, v_1, v_2, v_3])))
            sp_op += 1
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
            arg = st.unpack('<f', bytes(list([v_0, v_1, v_2, v_3])))
            sp_fl += 1
            steck_fl[sp_fl] = arg[0]
            ip += 4
            loger.debug(arg[0])
        # создаем одно ядро в массиве
        # пришла команда создать ядро
        elif op == make_kernel:
            try:
                loger.debug("-in make kernel-")
                layer = nn_params.net[n_lay]

                out = ops_st[sp_op]
                sp_op -= 1
                loger.debug(f"out {out}")

                in_ = ops_st[sp_op]
                sp_op -= 1
                loger.debug(f"in_ {in_}")

                layer.in_ = in_
                layer.out = out

                is_with_bias = ops_st[sp_op]
                sp_op -= 1
                with_bias = False
                if is_with_bias:
                    with_bias = True
                else:
                    with_bias = False
                layer.with_bias = with_bias

                what_func = ops_st[sp_op]
                sp_op -= 1
                layer.act_func = what_func
                
                # make_kernel_f(nn_params, net, n_lay, matrix_el_st, ops_st, sp_op)
                com_el_amount = in_ * out
                matrix = [[0, 0], [0, 0], [0, 0]]
                for row in range(out):
                    for elem in range(in_):
                        nn_params.net[n_lay].matrix[row][elem] =\
                            steck_fl[row * in_ + elem]
                        # matrix[row][elem]=steck_fl[row * in_ + elem]
                loger.debug(f'matrix {matrix}')
                sp_fl -= com_el_amount
                # переходим к следующему индексу ядра
                n_lay += 1
            except Exception as e:
                loger.debug(f'steck fl {steck_fl}')
                loger.debug(f'{e.args}')
                loger.debug(f'matrix {matrix}')
        # пришла команда узнать пользуемся ли биасами
        # надо извлечь параметр
        # elif op == with_bias:
        #     is_with_bias = ops_st[sp_op]
        #     sp_op -= 1
        #     if is_with_bias == 1:
        #         nn_params.with_bias = True
        #     elif is_with_bias == 0:
        #         nn_params.with_bias = False
        elif op == determe_alpha_and_beta_tan:
            beta = steck_fl[sp_fl]
            sp_fl -= 1
            alpha = steck_fl[sp_op]
            sp_fl -= 1
            nn_params.alpha_tan = alpha
            nn_params.beta_tan = beta
        elif op == determe_alpha_sigmoid:
            alpha = steck_fl[sp_op]
            sp_fl -= 1
            nn_params.alpha_sigmoid = alpha
        elif op == determe_alpha_leaky_relu:
            alpha = steck_fl[sp_op]
            sp_fl -= 1
            nn_params.alpha_leaky_relu = alpha
        # показываем на следующую инструкцию
        # loger.debug(f'steck fl {steck_fl}')
        ip += 1
        op = buffer[ip]

    # также подсчитаем сколько у наc ядер
    nn_params.nl_count = n_lay
    # находим количество входов
    # -1  # -1 зависит от биасов
    nn_params.input_neurons = nn_params.net[0].in_
    # находим количество выходов когда образовали сеть
    nn_params.outpu_neurons = nn_params.net[nn_params.nl_count-1].out


def deserialization(nn_params, fname: str, loger):
    buffer = [0] * max_spec_elems_1000
    buf_str = b''
    with open(fname, 'rb') as f:
        buf_str = f.read()
        if len(buf_str) >= len(buffer):
            print("Static memory error", end=' ')
            print("in buffer deserialization(buffer we read serealized net from file)")
            print("[init in serial_deserial.deserialization()]")
            loger.error(
                "Static memory error in buffer-deserialization (buffer we read serealized net from file)[init in serial_deserial.deserialization()]")
            return

    cn_by = 0
    for i in buf_str:
        buffer[cn_by] = i
        cn_by += 1
    # разборка байт-кода
    deserialization_vm(nn_params, buffer, loger)
# ----------------------------------------------------------------------
