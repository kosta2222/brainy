#NN_params.[py]
from .nn_constants import max_in_nn,max_trainSet_rows,max_validSet_rows,max_rows_orOut,\
    max_am_layer,max_am_epoch,max_am_objMse,max_stack_matrEl,max_stack_otherOp,bc_bufLen, NOP, SIGMOID
from .Lay import Lay
# Параметры сети
class  NN_params:
    def __init__(self):
        self.net=[]
        for i in range(max_am_layer):
            ob_lay=Lay()
            self.net.append(ob_lay)  # вектор слоев
        self.input_neurons=0  # количество выходных нейронов
        self.outpu_neurons=0  # количество входных нейронов
        self.nl_count=0  # количество слоев
        self.inputs=[0]*(max_in_nn)  # входа сети
        self.targets=[0]*(max_rows_orOut)  # ответы от учителя
        self.out_errors = [0] * (max_rows_orOut)  # вектор ошибок слоя
        self.lr=0;  # коэффициент обучения
        self.with_adap_lr = False
        self.with_bias = False
        self.act_fu = SIGMOID
        self.alpha_leaky_relu = 0.01
        self.alpha_sigmoid = 0.42
        self.alpha_tan = 1.7159
        self.beta_tan = 2 / 3
