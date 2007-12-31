from unittest import TestCase
from brainy.learn import cr_lay
from brainy.nn_constants import RELU, SOFTMAX
from brainy.NN_params import NN_params
from brainy.util import get_logger
from brainy.learn import make_hidden

loger, date= get_logger("debug", "new.log", __name__)

class TestCr_lay(TestCase):
    def test_cr_lay(self):
        i=0
        loger.info(date)
        nn_params=NN_params()
        i=cr_lay(loger, nn_params, 'D', 2, 3, SOFTMAX)
        make_hidden(nn_params,nn_params.net[i], [1, 1], loger)
        print("out_nn",nn_params.net[i].hidden)
        # self.fail()
