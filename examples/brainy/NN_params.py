#NN_params.[py]
from .nn_constants import max_in_nn_1000,max_trainSet_rows,max_validSet_rows,max_rows_orOut_10,\
    max_am_layer,max_am_epoch,max_am_objMse,max_stack_matrEl,max_stack_otherOp_10,bc_bufLen, NOP, SIGMOID
from .Lay import Lay, Dense
# Параметры сети
class  NN_params:
    def __init__(self):
        self.net=[]
        self.denses=[]
        for i in range(max_am_layer):
            ob_lay=Lay()
            self.net.append(ob_lay)  # вектор слоев
        for i in range(max_am_layer):
            self.denses.append(Dense())
        self.sp_d=-1
        self.input_neurons=0  # количество выходных нейронов
        self.outpu_neurons=0  # количество входных нейронов
        self.nl_count=0  # количество слоев
        self.inputs=[0]*(max_in_nn_1000 * 10)  # входа сети
        self.targets=[0]*(max_rows_orOut_10)  # ответы от учителя
        self.out_errors = [0] * (max_rows_orOut_10)  # вектор ошибок слоя
        self.lr=0;  # коэффициент обучения
        self.loss_func=NOP
        self.with_adap_lr = False
        self.with_bias = False
        self.act_fu = SIGMOID
        self.alpha_leaky_relu = 0.01
        self.alpha_sigmoid = 0.42
        self.alpha_tan = 1.7159
        self.beta_tan = 2 / 3
        self.mse_treshold = 0.001

    def __str__(self):
        b_codes = ['x', 'RELU', 'x', 'SIGMOID', 'x', 'TRESHHOLD_FUNC', 'x', 'LEAKY_RELU', 'x', 'TAN']
        func_s=b_codes[self.act_fu]
        ind=b_codes.index(func_s)
        act_fu=b_codes[ind]
        info=f'with-adap-lr: {self.with_adap_lr}\nwith-bias: {self.with_bias}\n'+\
             f'act-fu: {act_fu}\n'+\
             f'alpha-leaky-relu: {self.alpha_leaky_relu} alpha-sigmoid: {self.alpha_sigmoid} alpha-tan: {self.alpha_tan} beta-tan: {self.beta_tan}\n'+\
             f'mse-treshold: {self.mse_treshold}'
        return info
