from unittest import TestCase
from brainy.learn import cr_lay
from brainy.nn_constants import RELU, SOFTMAX
from brainy.NN_params import NN_params
from brainy.util import get_logger
from brainy.learn import make_hidden, feed_forwarding, feed_forwarding_on_contrary

loger, date= get_logger("debug", "new.log", __name__)

class TestCr_lay(TestCase):
    def test_cr_lay(self):
        i=0
        loger.info(date)
        nn_params=NN_params()
        i=cr_lay(loger, nn_params, 'D', 2, 3, SOFTMAX)
        # make_hidden(nn_params,nn_params.net[i], [1, 1], loger)
        nn_params.inputs=[1, 1]
        nn_params.with_bias=True
        # loger.debug(nn_params)
        print("out_nn",nn_params.net[i].hidden)
        feed_forwarding(nn_params, True, loger)
        loger.debug(f'hidden[:3] {nn_params.net[i].hidden[:3]}')
        loger.debug(nn_params)
        nn_params.net[i].hidden[:3]=(0, 0, 0)
        loger.info('\n---')
        nn_params.inputs=[0.333,0.333,0.333]
        feed_forwarding_on_contrary(nn_params,True,loger)
        loger.debug(nn_params)
        loger.debug(f'hidden_contr[:3] {nn_params.net[i].hidden[:3]}')
