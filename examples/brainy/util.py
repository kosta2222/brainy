#----------------------дебаг, хелп функции--------------------------------------
import logging
def get_logger(level):
    logger = None
    logger = logging.getLogger(__name__)
    if level == 'debug':
            logging.basicConfig(level=logging.DEBUG, filename='log.txt', filemode='w')
    elif level == 'release':
        logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w')
    return logger

def calc_list(list_:list):
    cn_elem = -1
    for i in range(len(list_)):
        elem=list_[i]
        cn_elem+=1
        if elem==0:
            print("i",i)
            print("elem",elem)
            print("op")
            break
        else:
            continue


    return cn_elem


def _0_(str_):
    print("Success ->", end = " ")
    print("function",str_)
