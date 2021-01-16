#-*-coding: cp1251-*-
from brainy.net_con_ import NetCon
from brainy.nn_constants import TRESHOLD_FUNC, PIECE_WISE_LINEAR, INIT_W_MY, SIGMOID, TAN,\
INIT_W_RANDN, INIT_W_RANDOM,\
TRESHOLD_FUNC_HALF, TAN, SIGMOID, LEAKY_RELU
import sys


def main():

    train_inp = ((1, 1), (0, 0), (0, 1), (1, 0))  # Логическое X
    train_out = ([0], [0], [1], [1])

    epochs = 10000
    l_r = 0.1

    errors_y = []
    epochs_x = []

    # Создаем обьект параметров сети

    net = NetCon(alpha_tan=1, beta_tan=1, alpha_sigmoid=1, alpha_leaky_relu=0.1)
    # Создаем слои
    net.cr_dense(2, 7,SIGMOID, True, INIT_W_MY)
    net.cr_dense(7, 1,SIGMOID, True, INIT_W_MY)
    net.get_b_c_bacward()
    print('net', net)
    try:
     for ep in range(epochs):  # Кол-во повторений для обучения
        gl_e = 0
        len_train_inp=len(train_inp)
        for single_array_ind in range(len_train_inp):

            inputs = train_inp[single_array_ind]
            output = net.feed_forwarding(inputs)

            

            net.backpropagate(train_out[single_array_ind],
                              train_inp[single_array_ind], l_r)
            e = net.calc_diff(output, train_out[single_array_ind])

            gl_e += net.get_err(e)                  
         
        
        print("error", gl_e)
        print("ep", ep)
        print()

        errors_y.append(gl_e)
        epochs_x.append(ep)

        if gl_e < 0.001:
              break

    
    except KeyboardInterrupt:   
      net.plot_gr('gr_py.png', errors_y, epochs_x)
      acc=net.evaluate(train_inp, train_out)
      print('acc', acc)
      sys.exit(0)

    net.plot_gr('gr_py.png', errors_y, epochs_x)
    acc=net.evaluate(train_inp, train_out) 
    print('acc', acc) 
    
    
main()