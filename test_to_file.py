from unittest import TestCase
from brainy.serial_deserial import to_file,deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import TAN, bc_bufLen, RELU
from brainy.learn import initiate_layers, answer_nn_direct
from brainy.fit import fit

X = []
Y = []
def create_nn_params():
    return NN_params()
def spec_conf_nn_this_for_this_prog(nn_in_amount, nn_out_amount):
    nn_params = create_nn_params()
    nn_params.with_bias = False
    nn_params.with_adap_lr = True
    nn_params.lr = 0.01
    nn_params.act_fu = RELU
    nn_params.alpha_sigmoid = 0.056
    nn_in_amount = 20
    nn_out_amount = 1
    nn_map = (nn_in_amount, 8, nn_out_amount)
    initiate_layers(nn_params, nn_map, len(nn_map))
    return nn_params
def calc_sent_vecs(str_x:str):
    nn_in_amount=20
    ord_as_devided_val = 0.0
    float_x = [0] * nn_in_amount
    str_y = [1]
    Y.append(str_y)
    cn_char = 0
    for chr in str_x:
        ord_as_devided_val = ord(chr) / 255
        float_x[cn_char]= round(ord_as_devided_val, 2)
        cn_char+= 1
    X.append(float_x)
class TestTo_file(TestCase):
    def setUp(self):
      self.nn_params=NN_params()
      self.nn_params_new = NN_params()
      self.buffer = [0] * bc_bufLen



    def test_to_file(self):
        for row in range(1):
            for elem in range(3):
                self.nn_params.net[0].matrix[row][elem] = 1
        self.nn_params.net[0].in_ = 3
        self.nn_params.net[0].out = 1
        print("net", self.nn_params.net[0].matrix)
        to_file(self.nn_params, self.buffer, self.nn_params.net, 1 , 'wei_test.my')
        deserialization(self.nn_params_new,self.nn_params_new.net, 'wei_test.my')
        print('in',self.nn_params_new.net[0].in_)
        print('out',self.nn_params_new.net[0].out)
        # self.fail()
    def test_to_file2(self):
        for row in range(2):
            for elem in range(3):
                self.nn_params.net[0].matrix[row][elem] = 1
        for row in range(1):
             for elem in range(2):
                self.nn_params.net[1].matrix[row][elem] = 1
        self.nn_params.net[0].in_ = 3
        self.nn_params.net[0].out = 2
        self.nn_params.net[1].in_ = 2
        self.nn_params.net[1].out = 1
        print("net", self.nn_params.net[0].matrix)
        to_file(self.nn_params, self.buffer, self.nn_params.net, 2 , 'wei_test.my')
        deserialization(self.nn_params_new,self.nn_params_new.net, 'wei_test.my')
        print('in',self.nn_params_new.net[0].in_)
        print('out',self.nn_params_new.net[0].out)
        print('in',self.nn_params_new.net[1].in_)
        print('out',self.nn_params_new.net[1].out)
        # self.fail()]
    def test_many(self):
        buffer_ser = [0] * bc_bufLen * 5
        nn_params = spec_conf_nn_this_for_this_prog(8, 1)
        nn_params_new=create_nn_params()
        calc_sent_vecs('lampu')
        fit(self.buffer, nn_params, 10, X, Y, X, Y, 100, use_logger='release')
        kernel_amount = nn_params.nl_count
        file_save = "weight_file.my"
        to_file(nn_params, buffer_ser, nn_params.net, kernel_amount, file_save)
        deserialization(nn_params_new, nn_params_new.net, file_save)
        answer=answer_nn_direct(nn_params_new,X[0],1)
        print(answer)
