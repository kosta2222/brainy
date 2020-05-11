from unittest import TestCase
from brainy.serial_deserial import to_file,deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import TAN, bc_bufLen



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
        # self.fail()
