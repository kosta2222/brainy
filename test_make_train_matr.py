from unittest import TestCase
from .util import make_train_matr
import numpy as np
# np.set_printoptions(threshold=np.inf)


class TestMake_train_matr(TestCase):
    def test_make_train_matr(self):
        p='b:/src'
        X=make_train_matr(p)
        print("X", X.tolist())
        print("X[0] i X[1]",np.array_equal(X[0],X[1]))
        print("X[0] shape",X[0].shape)
        print("X tolist 1",X.tolist()[2])
        # self.fail()
