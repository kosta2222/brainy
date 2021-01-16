import math
import numpy as np
import matplotlib.pyplot as plt
import math
import random
# from .serial_deserial import to_file, deserialization
import sys

from .nn_constants import *



class Dense:
    def __init__(self):  # конструктор
        self.in_ = None  # количество входов слоя
        self.out_ = None  # количество выходов слоя
        self.matrix = [0] * 100  # матрица весов
        self.biases = [0] * 100  # вектор биасов
        self.cost_signals = [0] * 100  # вектор взвешенного состояния нейронов
        self.act_func = RELU
        self.hidden = [0] * 100  # вектор после функции активации
        self.errors = [0] * 100  # вектор ошибок слоя
        self.biases_errors=[0]*100
        self.with_bias = False
        for row in range(100):  # создаем матрицу весов
            # подготовка матрицы весов,внутренняя матрица
            self.inner_m = list([0] * 100)
            self.matrix[row] = self.inner_m


################### Функции обучения ######################


class NetCon:
    def __init__(self, alpha_sigmoid=1, alpha_tan=1, beta_tan=1, alpha_leaky_relu=1):
        self.net_dense = [None] * 3
        self.alpha_sigmoid = alpha_sigmoid
        self.alpha_tan = alpha_tan
        self.beta_tan = beta_tan
        self.alpha_leaky_relu = alpha_leaky_relu
        for i in range(3):  # Статическое выделение слоев
            self.net_dense[i] = Dense()
        self.sp_d = -1  # алокатор для слоев fcn
        self.b_c_forward = []
        self.b_c_bacward_tmp = []
        self.b_c_bacward = None
        self.ip = 0
        self.ready = True

    def make_hidden(self, layer, inputs: list):
        len_layer_out = layer.out_
        len_layer_in = layer.in_
        for row in range(len_layer_out):
            summ = 0
            for elem in range(len_layer_in):
                summ += layer.matrix[row][elem] *\
                    inputs[elem]
            if layer.with_bias:  # сколько рядов столько и элементов в векторе биасов
                summ+=layer.biases[row]
            val = self.operations(layer.act_func, summ)
            layer.hidden[row] = val

    def get_hidden(self, objLay: Dense):
        return objLay.hidden

    def cr_dense(self,   in_=0, out_=0, act_func=TRESHOLD_FUNC_HALF, with_bias=True, init_w=INIT_W_MY):
        self.sp_d += 1
        layer = self.net_dense[self.sp_d]
        layer.in_ = in_
        layer.out_ = out_
        layer.act_func = act_func

        if with_bias:
            layer.with_bias = True
        else:
            layer.with_bias = False

        # инициализируем веса и биасы
        for row in range(out_):
            for elem in range(in_):
                layer.matrix[row][elem] = self.operations(
                    init_w,
                    0)
            if layer.with_bias:
                layer.biases[row] = self.operations(
                    init_w, 0)

        # просто байткод для прямого распространения
        self.b_c_forward.append(DENSE)
        self.b_c_forward.append(self.sp_d)
        # байткод будет наоборот
        self.b_c_bacward_tmp.append(self.sp_d)
        self.b_c_bacward_tmp.append(DENSE)

    # Различные операции по числовому коду

    def operations(self, op, x):

        if op == RELU:
            if (x <= 0):
                return 0
            else:
                return x
        elif op == RELU_DERIV:
            if (x <= 0):
                return 0
            else:
                return 1
        elif op == TRESHOLD_FUNC:
            if (x > 0):
                return 1
            else:
                return 0
        elif op == TRESHOLD_FUNC_HALF:
            if x >= 0.5:
                return 1
            else:
                return 0
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == PIECE_WISE_LINEAR:
            if x >= 1/2:
                return 1
            elif x < 1/2 and x > -1/2:
                return x
            elif x <= -1/2:
                return 0
        elif op == PIECE_WISE_LINEAR_DERIV:
            return 1
        elif op == TRESHOLD_FUNC_DERIV:
            return 1
        elif op == LEAKY_RELU:
            if (x <= 0):
                return self.alpha_leaky_relu
            else:
                return 1
        elif op == LEAKY_RELU_DERIV:
            if (x <= 0):
                return self.alpha_leaky_relu
            else:
                return 1
        elif op == SIGMOID:
            y = 1 / (1 + math.exp(-self.alpha_sigmoid * x))
            return y
        elif op == SIGMOID_DERIV:
            return self.alpha_sigmoid * x * (1 - x)
        elif op == INIT_W_MY:
            if self.ready:
                self.ready = False
                return 1  # -0.567141530112327
            self.ready = True
            return -1  # 0.567141530112327
        elif op == INIT_W_RANDOM:

            return random.random()
        elif op == TAN:
            y = self.alpha_tan * math.tanh(self.beta_tan * x)
            return y
        elif op == TAN_DERIV:
            c = self.beta_tan/self.alpha_tan
            return c * (self.alpha_tan**2 - x * x)
        elif op == INIT_W_CONST:
            return 0.567141530112327
        elif op == INIT_W_RANDN:
            return np.random.randn()
        else:
            print("Op or function does not support ", op)

    def get_b_c_bacward(self):
        """
        Сформировать байткод для обратного распространения
        """
        len_b_c_bacward = len(self.b_c_bacward_tmp)
        self.b_c_bacward = [0] * len_b_c_bacward
        for i in range(len_b_c_bacward):
            # из очереди в байткод
            self.b_c_bacward[i] = self.b_c_bacward_tmp.pop()

    def calc_out_error(self, layer, targets):
        out_ = layer.out_
        for row in range(out_):
            layer.errors[row] =\
                (layer.hidden[row] - targets[row]) * self.operations(
                layer.act_func + 1, layer.hidden[row])
            if layer.with_bias:
                layer.biases_errors[row]=layer.errors[row]

    def calc_hid_error(self, layer, layer_next):
        len_layer_next_in = layer_next.in_
        len_layer_next_out = layer_next.out_
        for elem in range(len_layer_next_in):
            summ = 0
            for row in range(len_layer_next_out):
                summ += layer_next.matrix[row][elem] * \
                    layer_next.errors[row]

            layer.errors[elem] = summ * self.operations(
                layer.act_func + 1, layer.hidden[elem])
            if layer.with_bias:
                layer.biases_errors[elem] = layer.errors[elem]

    def upd_matrix(self, layer, errors, inputs, lr):
        len_layer_out = layer.out_
        len_layer_in = layer.in_
        for row in range(len_layer_out):
            error = errors[row]
            error_bias=layer.biases_errors[row]
            for elem in range(len_layer_in):
                layer.matrix[row][elem] -= lr * \
                    error * inputs[elem]
                if layer.with_bias:
                    layer.biases[elem] -= error_bias * 1

    def calc_diff(self, out_nn, teacher_answ):
        diff = [0] * len(out_nn)
        for row in range(len(teacher_answ)):
            diff[row] = out_nn[row] - teacher_answ[row]
        return diff

    def get_err(self, diff):
        sum = 0
        for row in range(len(diff)):
            sum += diff[row] * diff[row]
        return sum / len(diff)

    def feed_forwarding(self, inputs):
        len_b_c_forward = len(self.b_c_forward)
        while self.ip < len_b_c_forward:
            op = self.b_c_forward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_forward[self.ip]

                if i == 0:
                    layer = self.net_dense[0]
                    self.make_hidden(layer, inputs)
                else:
                    layer = self.net_dense[i]
                    layer_prev = self.net_dense[i - 1]
                    self.make_hidden(layer, self.get_hidden(layer_prev))
            self.ip += 1

        self.ip = 0  # сбрасываем ip так прямое распространение будет в цикле

        last_layer = self.net_dense[self.sp_d]

        return self.get_hidden(last_layer)

    def backpropagate(self, y, x, l_r):
        len_b_c_bacward = len(self.b_c_bacward)

        while self.ip < len_b_c_bacward:
            op = self.b_c_bacward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                layer = self.net_dense[i]
                if i == self.sp_d:
                    self.calc_out_error(layer, y)
                else:
                    layer_next = self.net_dense[i + 1]
                    self.calc_hid_error(layer, layer_next)
            self.ip += 1

        self.ip = 0

        while self.ip < len_b_c_bacward:
            op = self.b_c_bacward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                layer = self.net_dense[i]
                layer_prev = self.net_dense[i - 1]
                if i == 0:
                    self.upd_matrix(self.net_dense[i], self.net_dense[i].errors,
                                    x, l_r)
                    # layer.errors=[0]*10
                else:
                    self.upd_matrix(layer, layer.errors,
                                    layer_prev.hidden, l_r)

            self.ip += 1

        self.ip = 0

    def answer_nn_direct(self, inputs):
        out_nn = self.feed_forwarding(inputs)
        return out_nn

    def evaluate(self, X_test, Y_test):
        """
         Оценка набора в процентах
         X_test: матрица обучающего набора X
         Y_test: матрица ответов Y
         return точность в процентах
        """
        scores = []
        res_acc = 0
        rows = len(X_test)
        wi_y_test = len(Y_test[0])
        elem_of_out_nn = 0
        elem_answer = 0
        is_vecs_are_equal = False
        for row in range(rows):
            x_test = X_test[row]
            y_test = Y_test[row]

            out_nn = self.answer_nn_direct(x_test)
            for elem in range(wi_y_test):
                elem_of_out_nn = out_nn[elem]
                elem_answer = y_test[elem]
                if elem_of_out_nn > 0.5:
                    elem_of_out_nn = 1
                    print("output vector elem -> ( %f ) " % 1, end=' ')
                    print("expected vector elem -> ( %f )" %
                          elem_answer, end=' ')
                else:
                    elem_of_out_nn = 0
                    print("output vector elem -> ( %f ) " % 0, end=' ')
                    print("expected vector elem -> ( %f )" %
                          elem_answer, end=' ')
                if elem_of_out_nn == elem_answer:
                    is_vecs_are_equal = True
                else:
                    is_vecs_are_equal = False
                    break
            if is_vecs_are_equal:
                print("-Vecs are equal-")
                scores.append(1)
            else:
                print("-Vecs are not equal-")
                scores.append(0)
        # print("in eval scores",scores)
        res_acc = sum(scores) / rows * 100

        return res_acc

    def plot_gr(self, f_name: str, errors: list, epochs: list) -> None:
        fig: plt.Figure = None
        ax: plt.Axes = None
        fig, ax = plt.subplots()
        ax.plot(epochs, errors,
                label="learning",
                )
        plt.xlabel('Эпоха обучения')
        plt.ylabel('loss')
        ax.legend()
        plt.savefig(f_name)
        print("Graphic saved")
        plt.show()

    def __str__(self):
        return "bc"+str(self.b_c_forward)+"rev b_c"+str(self.b_c_bacward)
#############################################


if __name__ == '__main__':

    def main():

        train_inp = ((1, 1), (0, 0), (0, 1), (1, 0))  # Логическое И
        train_out = ([0], [0], [1], [1])

        epochs = 5000
        l_r = 0.07

        errors_y = []
        epochs_x = []

        # Создаем обьект параметров сети

        net = NetCon()
        # Создаем слои
        net.cr_dense(2, 7, RELU, True, INIT_W_MY)
        net.cr_dense(7, 1, RELU, True, INIT_W_MY)
        net.get_b_c_bacward()
        for ep in range(epochs):  # Кол-во повторений для обучения
            gl_e = 0
            for single_array_ind in range(len(train_inp)):

                inputs = train_inp[single_array_ind]
                output = net.feed_forwarding(inputs)

                e = net.calc_diff(output, train_out[single_array_ind])

                gl_e += net.get_err(e)

                net.backpropagate(train_out[single_array_ind],
                                  train_inp[single_array_ind], l_r)

            # gl_e /= 2
            print("error", gl_e)
            print("ep", ep)
            print()

            errors_y.append(gl_e)
            epochs_x.append(ep)

            if gl_e == 0:
                break

        net.plot_gr('gr.png', errors_y, epochs_x)
        acc = net.evaluate(train_inp, train_out)
        print('acc', acc)

    main()
