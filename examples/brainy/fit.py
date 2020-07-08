from .cross_val_eval import evaluate
from .learn import train, get_min_square_err, get_mean
import datetime as d
"""
X и Y - означает матрицы обучения и ответов соответственно(массив с другими просто массивами)
x*_ и  y*_ - вектор из этих матриц(просто массив)
"""
def fit(nn_params, epochcs, X, Y, X_eval, Y_eval, logger):
    """
    X_eval и Y_eval нужны потому что X и Y могут быть 'сжаты', а проверять нужно на 'целых' матрицах
    """
    logger.info('-in fit-')
    iteration = 1
    out_nn=None
    x = None  # 1d вектор из матрицы обучения
    y = None  # 1d вектор из матрицы ответов от учителя
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    hei_Y = len(Y)
    error=0
    error_pr=0
    delta_error= 0
    lr = nn_params.lr
    logger.info(str(nn_params))
    exit_flag=False
    is_net_learning = True
    while is_net_learning:
        logger.info(f'=iteration= {iteration}')
        for i in range(hei_Y):
            x = X[i]
            y = Y[i]
            # logger.info(f'x: {x} y: {y}')
            train(nn_params, x, y, logger)
            out_nn = nn_params.net[nn_params.nl_count - 1].hidden
            mse = get_min_square_err(out_nn, y, nn_params.outpu_neurons)
            print(f"mse {round(mse,3)}")
            logger.info(f"mse {round(mse,3)}")
            acc = evaluate(nn_params, X_eval, Y_eval, logger)
            print(f'accuracy {round(acc,0)}')
            logger.info(f'accuracy {round(acc,0)}')
            if nn_params.with_loss_threshold:
              if round(acc,0) == round(nn_params.acc_shureness,0) and round(mse, 3) <= nn_params.mse_treshold:
                exit_flag=True
                break
        if exit_flag:
            break
        if not nn_params.with_loss_threshold:
            if epochcs==iteration:
                break
        iteration+=1
        if nn_params.with_adap_lr:
            error = get_mean(out_nn, y, len(y))
            delta_err = error - gama * error_pr
            if delta_err > 0:
               lr  = alpha * lr
            else:
               lr = beta * lr
            error_pr=error
            nn_params.lr = lr
        logger.info(f"learning rate {nn_params.lr}")
    logger.info("*CV (after batch)*")
    acc=evaluate(nn_params, X_eval, Y_eval, logger)
    logger.info(f'accuracy {acc}')
    return 0
    # compil_serializ(b_c, nn_params.net,len(nn_map)-1,"wei_wei")
