from brainy.NN_params import NN_params   # импортруем параметры сети
from brainy.serial_deserial import deserialization
from brainy.nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
from brainy.serial_deserial import to_file
from brainy.fit import fit
from brainy.learn import initiate_layers, answer_nn_direct
len_=10
push_i = 0
push_fl = 1
push_str = 2
calc_sent_vecs = 3
fit_ = 4
predict = 5
load = 6
stop = 7
X=[]
Y=[]
ops=["push_i","push_fl", "push_str", "calc_sent_vecs", "fit", "predict" ,"load"]
# X и Y означают двухмернй список обучения и ответов соответственно
# x_* и y_* - просто списки из этих матриц
# создать параметры сети
def create_nn_params():
    return NN_params()
def console(prompt, level):
        buffer = [0] * len_* 2  # байт-код для шелл-кода
        input_ = '<uninitialized>'
        # splitted_cmd и splitted_cmd_src - т.к. работаем со статическим массивом
        splitted_cmd: list = [''] * 2
        splitted_cmd_src:list = None
        main_cmd = '<uninitialized>'
        par_cmd = '<uninitialized>'
        idex_of_bytecode_is_bytecode = 0
        cmd_in_ops = '<uninitialized>'
        we_exit = 'exit'
        we_run = 'r'
        pos_bytecode = -1
        shell_is_running = True
        print("Здравствуйте я составитель кода этой программы")
        print("r - выполнить")
        print("exit - выход")
        print("Доступные коды:")
        for c in ops:
            print(c, end=' ')
        print()
        while shell_is_running:
            input_ = input(prompt)
            # полностью выходим из программы
            if input_== we_exit:
                break
            # выполняем байткод вирт-машиной
            elif input_== we_run:
                pos_bytecode+= 1
                buffer[pos_bytecode] = stop
                vm(buffer, level)
                pos_bytecode = -1
            splitted_cmd_src = input_.split()
            for pos_to_write in range(len(splitted_cmd_src)):
                splitted_cmd[pos_to_write] = splitted_cmd_src[pos_to_write]
            main_cmd = splitted_cmd[0]
            par_cmd = splitted_cmd[1]
            # Ищем код в списке код-строку
            for idex_of_bytecode_is_bytecode in range(len(ops)):
                cmd_in_ops = ops[idex_of_bytecode_is_bytecode]
                is_index_inside_arr = idex_of_bytecode_is_bytecode < len(ops)
                if  main_cmd == cmd_in_ops and is_index_inside_arr:
                    pos_bytecode += 1
                    # формируем числовой байт-код и если нужно значения параметра
                    buffer[pos_bytecode] = idex_of_bytecode_is_bytecode
                    if par_cmd != '':
                        pos_bytecode += 1
                        buffer[pos_bytecode] = par_cmd
                    # очищаем
                    splitted_cmd[0] = ''
                    splitted_cmd[1] = ''
                    break
def spec_conf_nn_this_for_this_prog(nn_in_amount, nn_out_amount):
   nn_params = create_nn_params()
   nn_params.with_bias = False
   nn_params.with_adap_lr = True
   nn_params.lr = 0.01
   nn_params.act_fu = RELU
   nn_params.alpha_sigmoid = 0.056
   nn_in_amount = 20
   nn_out_amount = 1
   nn_map = (nn_in_amount, 8, nn_out_amount)
   initiate_layers(nn_params, nn_map, len(nn_map))
   return nn_params
def vm(buffer:list, level):
    print("in vm")
    print("buff",buffer)
    nn_in_amount=20
    nn_out_amount=1
    nn_params = spec_conf_nn_this_for_this_prog(nn_in_amount, nn_out_amount)
    nn_params_new = create_nn_params()
    buffer_ser = [0] * bc_bufLen * 5  # буффер для сериализации матричных элементов и входов
    say_positive='Понятно. Постараюсь ваполнить вашу просьбу'
    say_negative='Извините ваша просьба неопознана'
    ip=0
    sp=-1
    sp_str=-1
    sp_fl=-1
    steck=[0]*len_
    steck_fl=[0.0]*len_
    steck_str=['']*len_
    op=0
    op=buffer[ip]
    while True:
        if op==push_i:
            sp+=1
            ip+=1
            steck[sp]=int(buffer[ip]) # Из строкового параметра
        elif op == push_fl:
            sp_fl += 1
            ip += 1
            steck_fl[sp_fl] = float(buffer[ip])  # Из строкового параметра
        elif op==push_str:
            sp_str+= 1
            ip += 1
            steck_str[sp_str] = buffer[ip]
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
        # elif op == calc_h_vecs:
        #     splited_par_x:list=None
        #     splited_par_y:list=None
        #     x = [0] * nn_in_amount
        #     y = [0] * nn_out_amount
        #     str_par_y = steck_str[sp_str]
        #     sp_str-= 1
        #     str_par_x = steck_str[sp_str]
        #     sp_str-= 1
        #     splited_par_x=str_par_x.split("_")
        #     splited_par_y=str_par_y.split("_")
        #     for i in range(len(splited_par_x)):
        #         x[i]=splited_par_x[i]
        #     for i in range(len(splited_par_y)):
        #         if splited_par_y[i]!='':
        #            y[i]=int(splited_par_y[i])
        #     X.append(x)
        #     Y.append(y)
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
           fit(buffer_ser, nn_params, 10, X_new_fix, Y_new_fix, X_new_fix, Y_new_fix, 100, use_logger=level)
           kernel_amount=nn_params.nl_count
           file_save="weight_file.my"
           to_file(nn_params, buffer_ser, nn_params.net,kernel_amount,file_save)
        elif op == predict:
            print("op predict")
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
                print("Сеть ответила ", nn_ans)
            else:
                print(say_negative)
                print("Сеть ответила ", nn_ans)
        elif op==load:
           print("op load")
           file_save = "weight_file.my"
           file_load = file_save
           deserialization(nn_params_new, nn_params_new.net, file_load)
        elif op == stop:
           return
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+= 1
        op = buffer[ip]
import sys
if __name__ == '__main__':
  if len(sys.argv)==2:
    level=sys.argv[1]
    if level == '-debug':
        pass
    elif level == '-release':
        pass
    else:
        print("Unrecognized option ",level)
        sys.exit(1)
    console('>>>', level)
  else:
      print("Program must have option: -release or -debug")
      sys.exit(1)

