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


def _0_(str_):
    print("Success ->", end = " ")
    print("function",str_)
