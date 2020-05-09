from unittest import TestCase
from brainy.serial_deserial import to_file,deserialization
from brainy.NN_params import NN_params
from brainy.nn_constants import TAN, bc_bufLen



class TestTo_file(TestCase):
    def setUp(self):
      self.nn_params=NN_params()
      for row in range(1):
          for elem in range(3):
             self.nn_params.net[0].matrix[row][elem]=1
      self.nn_params.net[0].in_=3
      self.nn_params.net[0].out=1
      self.nn_params.act_fu=TAN
      self.buffer = [0]*bc_bufLen
      self.nn_params_new=NN_params()
    def test_to_file(self):
        print("net", self.nn_params.net[0].matrix)
        to_file(self.nn_params, self.buffer, self.nn_params.net, 1 , 'wei_test.my')
        deserialization(self.nn_params_new,self.nn_params_new.net, 'wei_test.my')
        # self.fail()
