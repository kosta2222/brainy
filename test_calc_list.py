from unittest import TestCase
from .util import calc_list


class TestCalc_list(TestCase):
    def test_calc_list(self):
        list_=[1,3,4,5,0]
        val:int=0
        val=calc_list(list_)
        self.assertEqual(4, val)
        # self.fail("Ne doljno bit")
