from unittest import TestCase
from brainy.util import print_obj

class X:
    a=77
    b=88
    c=10
    l=[1, 2, 3, 4]
    d={1:3, 2:7}
    def __init__(self):
        self.t=12
        self.a=78
        self.l=self.l
        self.d=self.d
    def __str__(self):
        return print_obj('X', self.__dict__)
class TestPrint_obj(TestCase):
    def test_print_obj(self):
        o=X()
        # s=print_obj('X',o.__dict__)
        print(o)
        # self.fail()
