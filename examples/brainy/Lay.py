from .nn_constants import max_in_nn_1000, max_rows_orOut_10, max_spec_elems_1000, RELU
from .util import print_obj
class Lay:
    pass
class Dense:
    def __init__(self):# конструктор
        self.des='d'
        self.in_ = None  # количество входов слоя
        self.out = None  # количество выходов слоя
        self.matrix = list([])  # матрица весов
        self.cost_signals = [0] * max_rows_orOut_10 # вектор взвешенного состояния нейронов
        self.act_func=RELU
        self.hidden = [0] * max_rows_orOut_10  # вектор после функции активации
        self.errors = [0] * max_rows_orOut_10 # вектор ошибок слоя
        self.with_bias=False
        for row in range(max_rows_orOut_10):# создаем матрицу весов
            self.inner_m = list([0] * max_rows_orOut_10 * 10)  # подготовка матрицы весов,внутренняя матрица
            self.matrix.append(self.inner_m)

    #def __repr__(self):
        #return print_obj('Dense',self.__dict__)
