#-*-coding:utf-8-*-
#----------------------РґРµР±Р°Рі, С…РµР»Рї С„СѓРЅРєС†РёРё--------------------------------------
import logging
import numpy as np
import os
from PIL import Image
import datetime as d
from functools import wraps
from PIL import Image
import PIL
from os import listdir
import math


def get_logger(level_,fname,module,mode='w'):
    today=d.datetime.today()
    today_s=today.strftime('%x %X')
    logger = None
    logger = logging.getLogger(module)
    if level_ == 'debug':
            logging.basicConfig(level=logging.DEBUG, filename=fname, filemode=mode)

    elif level_ == 'release':
        logging.basicConfig(level=logging.INFO, filename=fname, filemode=mode)
    return logger, today_s


def convert_to_fur(data:list)->list:
    """
    Нахождение амплитуд сигнала
    data: сигнал
    return список 
    """
    n=len(data)
    dst=[None] * n
    matr=np.zeros((n, n))   
    i=0
    for row in range(n):
        k=0
        arg=2 * math.pi * i * k / n   
        for elem in range(n):   
            matr[row][elem]=math.cos(arg)
            k+=1
        i+=1   
    for row in range(n):
        tmp_v=0
        for elem in range(n):
            tmp_v+=matr[row][elem] * data[elem]
            tmp_v/=n
        dst[row]=tmp_v
    return dst 


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


def matr_img(path_:str,pixel_amount:int)->tuple:
    p=listdir(path_)
    # print("p",p)
    X=[]
    Y=[]
    fold_cont:list=None
    num_clss=len(p)
    img:PIL.Image=None
    data=None
    for fold_name_i in p:
        # print("fold_name_i",fold_name_i)
        fold_name_ind=int(fold_name_i.split('_')[0])
        p_tmp_ful=os.path.join(path_,fold_name_i)
        fold_content=listdir(p_tmp_ful)
        # print("fold_cont",fold_content)
        rows=len(fold_content)
        X_t=np.zeros((rows,pixel_amount))
        Y_t=np.zeros((rows,num_clss))
        f_index=0
        for file_name_j in fold_content:
            Y_t[f_index][fold_name_ind]=1
            img=Image.open(os.path.join(p_tmp_ful,file_name_j))
            data=list(img.getdata())
            X_t[f_index]=data
            # print("X_t[f_index]",X_t[f_index])
            print("file name",file_name_j)
            f_index+=1
        X_t=X_t.tolist()
        Y_t=Y_t.tolist()
        X.extend(X_t)
        Y.extend(Y_t)
        # print("X",X)
    return (X,Y)


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

