#----------------------дебаг, хелп функции--------------------------------------
import logging
import numpy as np
import os
from PIL import Image
import datetime as d
from functools import wraps
def get_logger(level_,fname,module):
    today=d.datetime.today()
    today_s=today.strftime('%x %X')
    logger = None
    logger = logging.getLogger(module)
    if level_ == 'debug':
            logging.basicConfig(level=logging.DEBUG, filename=fname, filemode='w')

    elif level_ == 'release':
        logging.basicConfig(level=logging.INFO, filename=fname, filemode='w')
    return logger, today_s

def calc_list(list_:list):
    cn_elem = 0
    for i in range(len(list_)):
        elem=list_[i]
        cn_elem+=1
        if elem==0:
            break
        else:
            continue
    return cn_elem


def calc_out_nn(l_: list):
    l_tested = [0] * 10000
    for i in range(len(l_)):
        val = round(l_[i], 1)
        if val > 0.5:
            l_tested[i] = 0
        else:
            l_tested[i] = 255
    return l_tested


def make_2d_arr(_1d_arr: list):
    matr_make = np.zeros(shape=(100, 100))
    for i in range(100):
        for j in range(100):
            matr_make[i][j] = _1d_arr[i * 100 + j]
    return matr_make


def make_train_matr(p_: str) -> np.ndarray:
    matr = np.zeros(shape=(4, 10000))
    data = None
    img = None
    cn_img = 0
    for i in os.listdir(p_):
        ful_p = os.path.join(p_, i)
        img = Image.open(ful_p)
        print("img", ful_p)
        data = list(img.getdata())
        matr[cn_img] = data
        cn_img+=1
    return matr


def _0_(str_):
    print("Success ->", end = " ")
    print("function",str_)
    return "Success ->function {}".format(str_)

def print_obj(name_obj_s,dict_obj:dict,si=50)->str:
    si=si
    res=''
    for k,v in dict_obj.items():
        if (not isinstance(v,int)) and (not isinstance(v, float)) and (not isinstance(v, bool)) and v!=None:
            assert('v_maybe_matrix','v_maybe_matrix')
            if len(v)>si or (isinstance(v[0], list) and len(v[0])>si):
               res+=k+' = '+ '<size of {0} [or list[0] ] is greater {1}>\n'.format(type(v), si)
               continue
        res+=k+' = '+str(v)
        res+='\n'
    return name_obj_s+':\n'+res

def tmp_wrap(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return tmp

