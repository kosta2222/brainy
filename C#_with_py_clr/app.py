import sys
import matplotlib.pyplot as plt
import clr
o = clr.AddReference("Program")
from Brainy import NetCon

TRESHOLD_FUNC = 0
TRESHOLD_FUNC_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11
INIT_W_CONST = 12
INIT_RANDN = 13
SOFTMAX = 14
SOFTMAX_DERIV = 15
PIECE_WISE_LINEAR = 16
PIECE_WISE_LINEAR_DERIV = 17
TRESHOLD_FUNC_HALF = 18
TRESHOLD_FUNC_HALF_DERIV = 19
MODIF_MSE = 20
DENSE = 21


def plot_gr(_file: str, errors: list, epochs: list) -> None:
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    ax.plot(epochs, errors,
            label="learning",
            )
    plt.xlabel('Эпоха обучения')
    plt.ylabel('loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    plt.show()


train_inp = ([1, 1], [0, 0], [0, 1], [1, 0])
train_out = ([0], [0], [1], [1])


def main():

    epochs = 100000
    l_r = 0.1

    errors_y = []
    epochs_x = []

    net = NetCon()
    # Создаем слои
    net.cr_dense(2, 3, TAN, True, INIT_W_MY)
    net.cr_dense(3, 1, SIGMOID, True, INIT_W_MY)
    net.end()

    for ep in range(epochs):  # Кол-во повторений для обучения
        gl_e = 0
        for single_array_ind in range(len(train_inp)):
            inputs = train_inp[single_array_ind]
            output = net.feed_forwarding(inputs)

            e = net.calc_diff(output, train_out[single_array_ind])
            net.backpropagate(train_out[single_array_ind],
                              inputs, l_r)
            gl_e += net.get_error(e)

        sys.stdout.write("\rIteration: {} and error {}".format(ep + 1, gl_e))
        errors_y.append(gl_e)
        epochs_x.append(ep)

        if gl_e < 0.001:
            break

    plot_gr('gr.png', errors_y, epochs_x)

    for single_array_ind in range(len(train_inp)):
        inputs = train_inp[single_array_ind]

        output_2_layer = net.feed_forwarding(inputs)

        equal_flag = 0
        out_nc = 1
        for row in range(out_nc):
            elem_net = output_2_layer[row]
            elem_train_out = train_out[single_array_ind][row]
            if elem_net > 0.5:
                elem_net = 1
            else:
                elem_net = 0
            print("elem:", elem_net)
            print("elem tr out:", elem_train_out)
            if elem_net == elem_train_out:
                equal_flag = 1
            else:
                equal_flag = 0
                break
        if equal_flag == 1:
            print('-vecs are equal-')
        else:
            print('-vecs are not equal-')

        print("========")


main()
