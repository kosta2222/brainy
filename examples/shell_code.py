import sys
import os
from brainy.NN_params import NnParams   # импортруем параметры сети
from brainy.serial_deserial_func import deserializ
from brainy.nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
from brainy.lear_func import initiate_layers, answer_nn_direct, answer_nn_direct_on_contrary
from brainy.serial_deserial_func import compil_serializ
from brainy.fit import fit
import numpy as np

# X и Y означают двухмернй список обучения и ответов соответственно
# x_* и y_* - просто списки из этих матриц

# создать параметры сети
def create_nn_params():
    return NnParams()
len_=10
push_i = 0
push_fl = 1
push_str = 2
calc_sent_vecs = 3
calc_h_vecs = 4
fit_ = 5
recogn = 6
what_say_positive = 7
load = 8
stop = 9
ops=["push_i","push_fl", "push_str", "calc_sent_vecs","calc_h_vecs","fit","recogn","what_say_positive","load"]
def console():
        b_c = [0] * len_* 3  # байт-код для шелл-кода
        input_ = '<uninitialized>'
        # splitted_cmd и splitted_cmd_src - т.к. работаем со статическим массивом
        splitted_cmd: list = [''] * 2
        splitted_cmd_src:list = None
        main_cmd = '<uninitialized>'
        par_cmd = '<uninitialized>'
        idex_of_bytecode_is_bytecode = 0
        cmd_in_ops = '<uninitialized>'
        pos_bytecode = -1
        shell_is_running = True
        is_we_didnt_faind_opcode = False
        print("Zdravstvuite ya sostavitel bait-coda dla etoi programmi")
        print("r vipolnit")
        print("Naberite exit dlya vihoda")
        print("Dostupnie codi:")
        for c in ops:
            print(c, end=' ')
        print()
        while shell_is_running:
            input_ = input(">>>")
            # полностью выходим из программы
            if input_== "exit":
                # exit_flag = True
                break
            # выполняем байткод вирт-машиной
            elif input_== "r":
                pos_bytecode+= 1
                b_c[pos_bytecode] = stop
                print("b_c",b_c)
                vm(b_c)
                pos_bytecode = -1
            splitted_cmd_src = input_.split()
            for pos_to_write in range(len(splitted_cmd_src)):
                splitted_cmd[pos_to_write] = splitted_cmd_src[pos_to_write]
            main_cmd = splitted_cmd[0]
            par_cmd = splitted_cmd[1]
            # Ищем код в списке код-строку
            for idex_of_bytecode_is_bytecode in range(len(ops)):
                cmd_in_ops = ops[idex_of_bytecode_is_bytecode]
                is_index_inside_arr = idex_of_bytecode_is_bytecode <= (len(ops) - 1)
                if cmd_in_ops == main_cmd and is_index_inside_arr:
                    pos_bytecode += 1
                    # формируем числовой байт-код и если нужно значения параметра
                    b_c[pos_bytecode] = idex_of_bytecode_is_bytecode
                    if par_cmd != '':
                        pos_bytecode += 1
                        b_c[pos_bytecode] = par_cmd
                    splitted_cmd[0] = ''
                    splitted_cmd[1] = ''
                    is_we_didnt_faind_opcode = False
                    break
                else :
                    is_we_didnt_faind_opcode =True
                    continue
                # Очищаем
            if is_we_didnt_faind_opcode:
                print("Izvintilyaus Net takogo opcoda")
                is_we_didnt_faind_opcode = False
X=[]
Y=[]
def spec_conf_nn_this_for_this_prog(nn_in_amount, nn_out_amount):
   nn_params = create_nn_params()
   nn_params.with_bias = True
   nn_params.with_adap_lr = True
   nn_params.lr = 0.01
   nn_params.act_fu = TAN
   nn_params.alpha_sigmoid = 0.056
   nn_in_amount = 20
   nn_out_amount = 1
   nn_map = (nn_in_amount, 8, nn_out_amount)
   initiate_layers(nn_params, nn_map, len(nn_map))
   return nn_params
def vm(b_c:list):
    nn_in_amount=20
    nn_out_amount=1
    nn_params = spec_conf_nn_this_for_this_prog(nn_in_amount, nn_out_amount)
    nn_params_new = create_nn_params()
    b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
    say_positive='<uninitialized>'
    say_negative='Izvinite vasha prosba ne opoznana'
    ip=0
    sp=-1
    sp_str=-1
    sp_fl=-1
    steck=[0]*len_
    steck_fl=[0.0]*len_
    steck_str=['']*len_
    op=0
    op=b_c[ip]
    while True:
        if op==push_i:
            sp+=1
            ip+=1
            steck[sp]=int(b_c[ip]) # Из строкового параметра
        elif op == push_fl:
            sp_fl += 1
            ip += 1
            steck_fl[sp_fl] = float(b_c[ip])  # Из строкового параметра
        elif op==push_str:
            sp_str+= 1
            ip += 1
            steck_str[sp_str] = b_c[ip]
        #  вычисление векторов это еще добавление к тренировочным матрицам
        elif op==calc_sent_vecs:
            ord_as_devided_val = 0.0
            float_x = [0] * nn_in_amount
            str_y = [1]
            Y.append(str_y)
            str_x=steck_str[sp_str]
            sp_str-=1
            cn_char = 0
            for chr in str_x:
                ord_as_devided_val = ord(chr) / 255
                float_x[cn_char]= round(ord_as_devided_val, 2)
                cn_char+= 1
            X.append(float_x)
            print("in vm in calc_ve:",X,Y)
        elif op == calc_h_vecs:
            splited_par_x:list=None
            splited_par_y:list=None
            x = [0] * nn_in_amount
            y = [0] * nn_out_amount
            str_par_y = steck_str[sp_str]
            sp_str-= 1
            str_par_x = steck_str[sp_str]
            sp_str-= 1
            splited_par_x=str_par_x.split("_")
            splited_par_y=str_par_y.split("_")
            for i in range(len(splited_par_x)):
                x[i]=splited_par_x[i]
            for i in range(len(splited_par_y)):
                if splited_par_y[i]!='':
                   y[i]=int(splited_par_y[i])
            X.append(x)
            Y.append(y)
        elif op==stop:
            return
        elif op == fit_:
           X_new_fix =[]
           Y_new_fix =[]
           x_new = [0] * nn_in_amount
           y_new = [0] * nn_out_amount
           for i in range(len(X)):
               X_new_fix.append(x_new)
           for i in range(len(Y)):
               Y_new_fix.append(y_new)
           for row in range(len(X)):
               for elem in range(nn_in_amount):
                   X_new_fix[row][elem] = X[row][elem]
           for row in range(len(Y)):
               for elem in range(nn_out_amount):
                   Y_new_fix[row][elem] = Y[row][elem]
           # X_new_fix_np=np.array(X_new_fix, dtype='float64')
           # Y_new_fix_np=np.array(Y_new_fix, dtype='float64')
           # X_new_fix_np-=np.mean(X_new_fix_np, dtype='float64', axis=0)
           # Y_new_fix_np-=np.mean(Y_new_fix_np, dtype='float64', axis=0)
           # X_new_fix_np=np.std(X_new_fix_np, axis = 0)
           # Y_new_fix_np=np.std(Y_new_fix_np, axis = 0)
           # print("in calc sent vecs X Y", X_new_fix_np, Y_new_fix_np)
           fit(b_c, nn_params, 10, X_new_fix, Y_new_fix, X_new_fix, Y_new_fix, 100)
           kernel_amount=len(nn_params.list_) - 1
           file_save="weight_file.my"
           compil_serializ(nn_params, b_c, nn_params.list_,kernel_amount,file_save)
        elif op == recogn:
            float_x = [0] * nn_in_amount
            str_x = steck_str[sp_str]
            sp_str-= 1
            cn_char = 0
            for chr in str_x:
                ord_as_devided_val = ord(chr) / 255
                float_x[cn_char] = round(ord_as_devided_val, 2)
                cn_char+=1
            nn_ans = answer_nn_direct(nn_params_new, float_x, 1)
            if nn_ans[0] >  0.559837 and nn_ans[0] <= 1:
                print(say_positive)
                print("nn answered", nn_ans)
            else:
                print(say_negative)
                print("nn answered", nn_ans)
        elif op==what_say_positive:
            say_str=steck_str[sp_str]
            sp_str-=1
            say_positive = say_str
            print("Pri negativnom skaju:",end=' ')
            print(say_negative)
        elif op==load:
           file_save = "weight_file.my"
           file_load = file_save
           deserializ(nn_params_new, nn_params_new.list_, file_load)
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+= 1
        op = b_c[ip]
if __name__ == '__main__':
    console()
