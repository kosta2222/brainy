from NN_params import NnParams   # импортруем параметры сети
from serial_deserial_func import deserializ
from nn_constants import bc_bufLen, RELU, LEAKY_RELU, SIGMOID, TAN
from lear_func import initiate_layers, answer_nn_direct, answer_nn_direct_on_contrary
from serial_deserial_func import compil_serializ
from fit import fit
import unittest as u
import numpy as np
#----------------------Основные параметры сети----------------------------------
# создать параметры сети
def create_nn_params():
    return NnParams()
class TestLay(u.TestCase):
    # Обучаю логическим операциям ИЛИ в test_1 и И в test_2
    # Если закоментирую один из тестов все проходит, если 2 сразу обучение останавливается на одном обучении
    # Или отдельно нажимаю на стрелочки testCase-ов в IDE - проходит
    def test_1(self):
        # устанавливаю параметры
        nn_params = create_nn_params()
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.act_fu = TAN
        nn_map = (2, 3, 1)
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y_or = [[1], [1], [1], [0]]
        X_gist = [[0, 1/4]]
        Y_gist = [[1/3]]
        b_c = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(nn_params, nn_map, len(nn_map))
        fit(b_c, nn_params, 7, X_gist, Y_gist, X, Y_or, 100)
        print("in test_1 after learn. matr")
        for i in nn_params.list_:
            print(i.matrix)
        # сериализуем
        compil_serializ(nn_params, b_c, nn_params.list_, len(nn_map) - 1, "weight_file.my" )
        # десериализуем
        nn_params_new = create_nn_params()
        deserializ(nn_params_new, nn_params_new.list_, "weight_file.my")
        print("in test_1 after deserializ. matr")
        for i in nn_params_new.list_:
            print(i.matrix)
        # предсказание
        print(answer_nn_direct(nn_params_new, [1, 1], 1))
        # предсказание наоборот
        print("*ON CONTRARY*")
        answer_nn_direct_on_contrary(nn_params_new, [0], 1)
        print("-------------")
    def test_2(self):
        # устанавливаю параметры
        nn_params = create_nn_params()
        nn_params.with_bias = False
        nn_params.with_adap_lr = True
        nn_params.lr = 0.01
        nn_params.act_fu = RELU
        nn_params.alpha_sigmoid= 0.056
        nn_map = (2, 3, 1)
        X = [[1, 1], [1, 0], [0, 1], [0, 0]]
        Y_and = [[1], [1], [1], [0]]
        X_gist = [[0, 1/4]]
        Y_gist = [[1]]
        b_c_new = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(nn_params, nn_map, len(nn_map))
        fit(b_c_new, nn_params, 7, X_gist, Y_gist, X, Y_and, 100)
        print("in test_1 after learn. matr")
        for i in nn_params.list_:
            print(i.matrix)
        # сериализуем
        compil_serializ(nn_params, b_c_new, nn_params.list_, len(nn_map) - 1, "weight_file" )
        # десериализуем
        nn_params_new = create_nn_params()
        deserializ(nn_params_new, nn_params_new.list_, "weight_file")
        print("in test_1 after deserializ. matr")
        for i in nn_params_new.list_:
            print(i.matrix)
        # предсказание
        print(answer_nn_direct(nn_params_new, [1, 1], 1))
        # предсказание наоборот
        print("*ON CONTRARY*")
        answer_nn_direct_on_contrary(nn_params_new, [0], 1)
        print("-------------")
    def test_3(self):
        # устанавливаю параметры
        nn_params = create_nn_params()
        nn_params.with_bias = True
        nn_params.with_adap_lr = False
        nn_params.lr = 0.01
        nn_params.act_fu = RELU
        nn_params.alpha_leaky_relu = 0.01
        nn_params.alpha_sigmoid= 0.056
        nn_map = (3, 2, 1)
        X = [[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1],[0, 1, 0]]
        Y = [[0], [1], [1], [1],[0]]
        X_np=np.array(X, dtype='float64')
        Y_np=np.array(Y, dtype='float64')
        X_np-=np.mean(X_np, axis=0, dtype='float64')
        Y_np-=np.mean(Y_np, axis=0, dtype='float64')
        X_np=[np.std(X_np, axis=0)]
        Y_np=[np.std(Y_np, axis=0)]
        X_my = [[1, 1/3, 1/3]]
        Y_my = [[1/3]]
        b_c_new = [0] * bc_bufLen  # буффер для сериализации матричных элементов и входов
        initiate_layers(nn_params, nn_map, len(nn_map))
        fit(b_c_new, nn_params, 7, X_np, Y_np, X, Y, 100)
        print("in test_1 after learn. matr")
        for i in nn_params.list_:
            print(i.matrix)
        # сериализуем
        compil_serializ(nn_params, b_c_new, nn_params.list_, len(nn_map) - 1, "weight_file" )
        # десериализуем
        nn_params_new = create_nn_params()
        deserializ(nn_params_new, nn_params_new.list_, "weight_file")
        print("in test_1 after deserializ. matr")
        for i in nn_params_new.list_:
            print(i.matrix)
        # предсказание
        # print(answer_nn_direct(nn_params_new, [1, 1], 1))
        # предсказание наоборот
        print("*ON CONTRARY*")
        # answer_nn_direct_on_contrary(nn_params_new, [0], 1)
        print("-------------")
if __name__ == '__main__':
    t = TestLay()
    u.main()
