from brainy.NN_params import NN_params   # импортруем параметры сети
from brainy.serial_deserial import deserialization
from brainy.nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN, SOFTMAX, MODIF_MSE, CROS_ENTROPY
from brainy.serial_deserial import to_file
from brainy.fit import fit
from brainy.learn import initiate_layers, answer_nn_direct, answer_nn_direct_on_contrary, cr_lay
from brainy.util import make_train_matr, make_2d_arr, calc_out_nn, get_logger, matr_img
import numpy as np
from PIL import Image
import logging
len_=10
stop=0
push_i = 1
push_fl = 2
push_str = 3
push_obj = 4
fit_ = 5
predict = 6
load = 7
fit_net=10
get_mult_class_matr=11
determe_X_Y=12
make_net=13
nparray=14
determe_X_eval_Y_eval=15

def create_nn_params():
    return NN_params()
def exect(buffer:list, loger:logging.Logger, date:str)->None:
    nn_params_new=create_nn_params()
    X_t=None
    Y_t=None
    X_eval=None
    Y_eval=None
    ip=0
    sp=-1
    steck=[0]*len_
    op=0
    op=buffer[ip]
    loger.info(f'Log started at {date}')
    while True:
        #------------основные коды памяти---------------
        if op==push_i:
            sp+=1
            ip+=1
            steck[sp]=int(buffer[ip]) # Из строкового параметра
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])  # Из строкового параметра
        elif op==push_str:
            sp+= 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op== push_obj:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        #----------------------------------------------
        # make_net 1-тип 2-тип слоя 3-количества нейронов 4-активационные функции 5-использовать ли биасы
        elif op==make_net:
           nn_params = create_nn_params() 
           l_tmp=None 
           acts_di={'s':SIGMOID,'r':RELU,'t':TAN,'SO':SOFTMAX,'l':LEAKY_RELU}
           ip += 1
           arg = buffer[ip]
           type_m, denses, inps, acts, use_bi = arg
           if type_m=='S':
               pass
           for i in range(len(denses)):
               if denses[i] == 'D':
                  splt_bi = use_bi[i].split('_')
                  if splt_bi[-1] == '1':
                     use_bias_ = True
                  elif splt_bi[-1] == '0':
                     use_bias_ = False
                  nn_params.with_bias=use_bias_
                  nn_params.input_neurons=inps[0]
                  nn_params.outpu_neurons=inps[-1]
                  l_tmp=cr_lay(nn_params, 'D', inps[i],inps[i+1], acts_di.get(acts[i]), loger)
        elif op==determe_X_Y:
            Y_t=steck[sp]
            sp-=1
            X_t=steck[sp]
            sp-=1
        elif op == determe_X_eval_Y_eval:
            Y_eval = steck[sp]
            sp -= 1
            X_eval = steck[sp]
            sp -= 1
        elif op==nparray:
            st_arg=steck[sp]
            sp-=1
            sp+=1
            steck[sp]=np.array(st_arg)
        # fit_net 1-эпохи 2-lr  3-использовать ли адаптивный lr 4-использовать ли пороговое обучение 5-mse-treshold 6-acc-shureness 7-loss-func
        elif op==fit_net:
            loss_func_di={"mse":MODIF_MSE,"crossentropy":CROS_ENTROPY}
            ip+=1
            arg=buffer[ip]
            eps,lr, is_wi_adap_lr,is_with_loss_threshold, mse_treshold, acc_shureness,loss_func=arg
            nn_params.with_loss_threshold=is_with_loss_threshold
            nn_params.with_adap_lr=is_wi_adap_lr
            nn_params.mse_treshold=mse_treshold
            nn_params.acc_shureness=acc_shureness
            nn_params.lr=lr
            nn_params.loss_func=loss_func_di.get(loss_func)
            if not X_eval and not Y_eval:
                X_eval=X_t
                Y_eval=Y_t
            print("X_t",X_t)
            print("Y_t",Y_t)
            print("X_eval",X_eval)
            print("Y_eval",Y_eval)
            fit(nn_params, eps, X_t, Y_t, X_eval, Y_eval, loger )
            to_file(nn_params, nn_params.net, loger, "to_file.my")
        elif op==get_mult_class_matr:
            pix_am=steck[sp]
            sp-=1
            path_=steck[sp]
            sp-=1
            X_t, Y_t=matr_img(path_,pix_am)
            X_t=np.array(X_t)
            Y_t=np.array(Y_t)
            X_t.astype('float32')
            Y_t.astype('float32')
            X_t/=255    
        elif op == predict:
            deserialization(nn_params_new,"to_file.my",loger)
            out_nn=answer_nn_direct(nn_params_new, X_t, loger)
            print("answer",out_nn)
        elif op == stop:
           return
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+= 1
        op=buffer[ip]

import sys
if __name__ == '__main__':
  loger=None
  if len(sys.argv)==2:
    level=sys.argv[1]
    print("level",level)
    if level == '-debug':
        loger, date=get_logger("debug", 'log_cons.log', __name__)
    elif level == '-release':
        loger, date=get_logger("release", 'log_cons.log', __name__)
    else:
        print("Unrecognized option ",level)
        sys.exit(1)
    p1=(push_str,r'B:\msys64\home\msys_u\code\python\brainy\examples\train_ann',push_i,784,get_mult_class_matr,
        push_i,15,push_fl,100,fit_net,
        push_str,r'B:\msys64\home\msys_u\code\python\brainy\examples\ask_ann',push_i,784,get_mult_class_matr,
        predict,
        stop)

# fit_net 1-эпохи 2-lr  3-использовать ли адаптивный lr 4-использовать ли пороговое обучение 5-mse-treshold 6-acc-shureness 7-loss-func
    X_and=[[0,1],[1,0],[0,0],[1,1]]
    Y_and=[[0],[0],[0],[1]]
    p2=(push_obj,X_and,push_obj,Y_and,determe_X_Y,
        make_net,('S', ('D','D','D'),(2,3,3,1),('r','r','t'),('usebias_0','usebias_0','usebias_0')),
        fit_net,(100,0.0001,False,True,0.0001,100,"mse"),
        stop)
    p3=(push_obj,[0,1],push_obj,[[None]],determe_X_Y,
        predict,
        stop)
    exect(p3,loger, date)
  else:
      print("Program must have option: -release or -debug")
      sys.exit(1)

