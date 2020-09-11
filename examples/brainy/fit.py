from .cross_val_eval import evaluate_new
from .learn import backpropagate, get_mse, feed_forwarding, get_mean, get_err, calc_diff
from .NN_params import Nn_params
from .util import convert_to_fur
import matplotlib.pyplot as plt


def plot_gr(_file: str, errors: list, epochs: list, name_gr: str, logger) -> None:
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    # plt.text(0.1, 1.1, name_gr)
    ax.plot(epochs, errors,
            label="learning",
            )
    # plt.plot(errors,label="errors")
    # plt.plot(history.history['acc'],label="Доля верных ответов на обучающем наборе")
    # if 'val_acc' in history.history: # Если работаем с val_acc
    #     plt.plot(history.history['val_acc'],label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    # logger.info("Graphic saved")
    plt.show()


def fit(nn_params: Nn_params, X, Y, X_test, Y_test, eps, l_r_, with_adap_lr, with_loss_threshold, ac_, mse_, loger):
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    error_pr = 0
    delta_error = 0
    l_r = l_r_
    net_is_running = True
    it = 0
    exit_flag = False
    mse = 0
    out_nn = None
    eps_l = []
    errs_l = []

    #loger.info(f'Log Started: {date}')

    while net_is_running:
        print("ep:", it)
        loger.info(f'ep: {it}')
        error = 0
        for retrive_ind in range(len(X)):
            x = X[retrive_ind]
            # print(f'x prost: {x}', end=' ')
            #x = convert_to_fur(x)
            # print(f'x fur: {x}')
            # x = np.array(x)
            y = Y[retrive_ind]
            #y = convert_to_fur(y)
            out_nn = feed_forwarding(nn_params, x, loger)
            error += get_err(calc_diff(out_nn, y, nn_params.outpu_neurons))
            # if with_adap_lr:
            #     delta_error = error - gama * error_pr
            #     if delta_error > 0:
            #         l_r = alpha * l_r
            #     else:
            #         l_r = beta * l_r
            #     error_pr = error
            backpropagate(nn_params, out_nn, y, x, 0.01, loger)
            ac = evaluate_new(nn_params, X_test, Y_test, loger)
            print("acc", ac)

        if with_adap_lr:
            delta_error = error - gama * error_pr
            if delta_error > 0:
                l_r = alpha * l_r
            else:
                l_r = beta * l_r
            error_pr = error            

        if exit_flag:
            break
        print("err", error)
        eps_l.append(it)
        errs_l.append(error)
        it += 1
        if with_loss_threshold:
            if error == mse_:
                break
        else:
            if it == eps:
                break

    plot_gr('gr.png', errs_l, eps_l, 'test', None)
