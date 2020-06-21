from .cross_val_eval import evaluate
from .learn import train, get_min_square_err, get_mean
import datetime as d
"""
X и Y - означает матрицы обучения и ответов соответственно(массив с другими просто массивами)
x*_ и  y*_ - вектор из этих матриц(просто массив)
"""
def fit(nn_params, epochcs, X:list, Y:list, X_eval:list, Y_eval, accuracy_eval_shureness:int, logger):
    """
    X_eval и Y_eval нужны потому что X и Y могут быть 'сжаты', а проверять нужно на 'целых' матрицах
    """
    today=d.datetime.today()
    today_s=today.strftime('%x %X')
    logger.info('-in fit-')
    iteration: int = 0
    A = nn_params.lr
    out_nn:list=None
    x:list = None  # 1d вектор из матрицы обучения
    y:list = None  # 1d вектор из матрицы ответов от учителя
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    hei_Y = len(Y)
    E_spec = 0
    logger.info( '--------------')
    logger.info(f'| Log started > {today_s}')
    logger.info( '--------------')
    print(today_s)
    print(str(nn_params))
    logger.info(str(nn_params))

    is_net_learning = True
    while is_net_learning:
        logger.info(f'iteration {iteration}')
        for i in range(hei_Y):
            x = X[i]
            y = Y[i]
            logger.info(f'x: {x} y: {y}')
            train(nn_params, x, y, logger)
            out_nn = nn_params.net[nn_params.nl_count - 1].hidden
            if nn_params.with_adap_lr:
                if iteration == 0:
                    E_spec_t_minus_1 = E_spec
                    A_t_minus_1 = A
                E_spec = get_mean(out_nn, y, len(y))
                delta_E_spec = E_spec - gama * E_spec_t_minus_1
                if delta_E_spec > 0:
                    A = alpha * A_t_minus_1
                else:
                    A = beta * A_t_minus_1
                A_t_minus_1 = A
                E_spec_t_minus_1 = E_spec
            nn_params.lr = A
            logger.debug(f"learning rate {A}")
        mse = get_min_square_err(out_nn, y, nn_params.outpu_neurons)
        print("mse:",mse)
        logger.info(f"mse {mse}")
        acc = evaluate(nn_params, X_eval, Y_eval, logger)
        print("accuracy:",acc)
        logger.info(f'accuracy {acc}')
        if acc == accuracy_eval_shureness and mse <= nn_params.mse_treshold:
            break
        iteration+=1
    logger.info("*CV (after batch)*")
    acc=evaluate(nn_params, X_eval, Y_eval, logger)
    logger.info(f'accuracy {acc}')
    return 0
    # compil_serializ(b_c, nn_params.net,len(nn_map)-1,"wei_wei")
