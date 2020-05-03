from .cross_val_eval import evaluate
from .learn import train, initiate_layers, get_min_square_err, answer_nn_direct, answer_nn_direct_on_contrary,\
get_mean
"""
X и Y - означает матрицы обучения и ответов соответственно(массив с другими просто массивами)
x*_ и  y*_ - вектор из этих матриц(просто массив)
"""
def fit(b_c:list, nn_params, epochcs, X:list, Y:list, X_eval:list, Y_eval, accuracy_eval_shureness:int):
    """
    X_eval и Y_eval нужны потому что X и Y могут быть 'сжаты', а проверять нужно на 'целых' матрицах
    """
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
    while True:
        for i in range(hei_Y):
            x = X[i]
            y = Y[i]
            train(nn_params, x, y, 1)
            out_nn = nn_params.net[nn_params.nlCount - 1].hidden
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
            mse = get_min_square_err(out_nn, y, nn_params.outputNeurons)
            print("in learn mse",mse)
        acc = evaluate(nn_params, X_eval, Y_eval)
        if acc == accuracy_eval_shureness and mse < 0.001:
            break
        iteration+=1
    print("***CV***")
    evaluate(nn_params, X_eval, Y_eval)
    # compil_serializ(b_c, nn_params.net,len(nn_map)-1,"wei_wei")
