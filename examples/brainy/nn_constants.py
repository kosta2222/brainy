# nn_constants.[py]
# Параметры статических массивов,количества слоев,количества эпох
max_in_nn = 10000
max_trainSet_rows = 4
max_validSet_rows = 10
max_rows_orOut = 20
max_am_layer = 7
max_am_epoch = 25
max_am_objMse = max_am_epoch
max_stack_matrEl = 256
max_stack_otherOp = 4
bc_bufLen = 256
elems_of_img=10000
max_spec_elems=10000
# команды для operations
RELU = 1
RELU_DERIV = 2
SIGMOID = 3
SIGMOID_DERIV = 4
TRESHOLD_FUNC = 5
TRESHOLD_FUNC_DERIV = 6
LEAKY_RELU = 7
LEAKY_RELU_DERIV = 8
INIT_W_HE = 9
INIT_W_GLOROT_MY = 10
DEBUG = 11
DEBUG_STR = 12
INIT_W_HABR = 13
INIT_W_MY = 14
INIT_W_UNIFORM = 15
TAN =16
TAN_DERIV =17
NOP = 18
INIT_W_HE_MY = 19
# байт-коды для сериализации/десериализации-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра, есть ли биасы,остановка ВМ
push_i = 1
push_fl = 2
make_kernel = 3
with_bias = 4
determe_act_func = 5
determe_alpha_leaky_relu = 6
determe_alpha_sigmoid = 7
determe_alpha_and_beta_tan = 8
determe_in_out = 9
stop = 10
