from .nn_constants import max_in_nn_1000, max_rows_orOut_10, max_spec_elems_1000
# Слой сети
class Lay:
    # in_ = None# количество входов слоя
    # out = None# количество выходов слоя
    def __init__(self):# конструктор
        self.in_ = None  # количество входов слоя
        self.out = None  # количество выходов слоя
        self.matrix = list([])  # матрица весов
        self.cost_signals = [0] * max_spec_elems_1000 *10 # вектор взвешенного состояния нейронов
        self.hidden = [0] * max_spec_elems_1000 * 10  # вектор после функции активации
        self.errors = [0] * max_spec_elems_1000 * 10 # вектор ошибок слоя
        for row in range(max_rows_orOut_10):# создаем матрицу весов
            self.inner_m = list([0] * (max_in_nn_1000*10))  # подготовка матрицы весов,внутренняя матрица
            self.matrix.append(self.inner_m)
