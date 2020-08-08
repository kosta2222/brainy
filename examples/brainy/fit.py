from .cross_val_eval import evaluate_new
from .learn import backpropagate, get_mse, feed_forwarding
from .NN_params import Nn_params
from .util import convert_to_fur
"""
X и Y - означает матрицы обучения и ответов соответственно(массив с другими просто массивами)
x*_ и  y*_ - вектор из этих матриц(просто массив)
"""
# def fit(nn_params, epochcs, X, Y, X_eval, Y_eval, logger):
#     """
#     X_eval и Y_eval нужны потому что X и Y могут быть 'сжаты', а проверять нужно на 'целых' матрицах
#     """
#     logger.info('-in fit-')
#     iteration = 1
#     out_nn=None
#     x = None  # 1d вектор из матрицы обучения
#     y = None  # 1d вектор из матрицы ответов от учителя
#     alpha = 0.99
#     beta = 1.01
#     gama = 1.01
#     hei_Y = len(Y)
#     error=0
#     error_pr=0
#     delta_error= 0
#     lr = nn_params.lr
#     logger.info(str(nn_params.net))
#     exit_flag=False
#     is_net_learning = True
#     while is_net_learning:
#         logger.info(f'=iteration= {iteration}')
#         for i in range(hei_Y):
#             x = X[i]
#             y = Y[i]
#             # logger.info(f'x: {x} y: {y}')
#             train(nn_params, x, y, logger)
#             out_nn = nn_params.net[nn_params.nl_count - 1].hidden
#             mse = get_min_square_err(out_nn, y, nn_params.outpu_neurons)
#             print(f"mse {round(mse,3)}")
#             logger.info(f"mse {round(mse,3)}")
#             acc = evaluate(nn_params, X_eval, Y_eval, logger)
#             print(f'accuracy {round(acc,0)}')
#             logger.info(f'accuracy {round(acc,0)}')
#             if nn_params.with_loss_threshold:
#               if round(acc,0) == round(nn_params.acc_shureness,0) and round(mse, 3) <= round(nn_params.mse_treshold, 3):
#                 exit_flag=True
#                 break
#         if exit_flag:
#             break
#         if not nn_params.with_loss_threshold:
#             if epochcs==iteration:
#                 break
#         iteration+=1
#         if nn_params.with_adap_lr:
#             error = get_min_square_err(out_nn, y, len(y))
#             delta_err = error - gama * error_pr
#             if delta_err > 0:
#                lr  = alpha * lr
#             else:
#                lr = beta * lr
#             error_pr=error
#             nn_params.lr = lr
#         logger.info(f"learning rate {nn_params.lr}")
#     logger.info("*CV (after batch)*")
#     acc=evaluate(nn_params, X_eval, Y_eval, logger)
#     logger.info(f'accuracy {acc}')
#     return 0
    # compil_serializ(b_c, nn_params.net,len(nn_map)-1,"wei_wei")
def fit(nn_params: Nn_params, X, Y, X_test, Y_test, eps, l_r_, with_adap_lr, ac_, mse_, loger):
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    error = 0
    error_pr = 0
    delta_error = 0
    l_r = l_r_
    net_is_running = True
    it = 0
    exit_flag = False
    mse = 0
    out_nn = None

    #loger.info(f'Log Started: {date}')

    while net_is_running:
        print("ep:", it)
        loger.info(f'ep: {it}')
        for retrive_ind in range(len(X)):
            x = X[retrive_ind]
            print(f'x prost: {x}', end=' ')
            #x = convert_to_fur(x)
            print(f'x fur: {x}')
            # x = np.array(x)
            y = Y[retrive_ind]
            #y = convert_to_fur(y)
            out_nn = feed_forwarding(nn_params, x, loger)
            loger.debug(f'out_nn: {out_nn}')
            mse = get_mse(out_nn, y, nn_params.outpu_neurons)
            # mse=get_cros_entropy(out_nn, y , nn_params.outpu_neurons)
            print("mse", mse)
            loger.info(f'mse: {mse}')
            print("out nn", out_nn)
            error = get_mse(out_nn, y, nn_params.outpu_neurons)
            if with_adap_lr:
                delta_error = error - gama * error_pr
                if delta_error > 0:
                    l_r = alpha * l_r
                else:
                    l_r = beta * l_r
                error_pr = error
            l_r*=0.0001 * it    
            backpropagate(nn_params, out_nn, y, x, l_r, loger)
            print("lr", l_r)
            ac = evaluate_new(nn_params, X_test, Y_test, loger)
            print("acc", ac)
            loger.info(f'acc: {ac}')
            if nn_params.with_loss_threshold:
                if ac == float(ac_) and mse < mse_:
                    exit_flag = True
                    break
        if exit_flag:
            break
        if not nn_params.with_loss_threshold:
         if it == eps:
             break
        it += 1
