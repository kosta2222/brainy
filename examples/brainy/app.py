import logging
import datetime as d


def get_logger(level_, fname, module, mode='w'):
    today = d.datetime.today()
    today_s = today.strftime('%x %X')
    logger = None
    logger = logging.getLogger(module)
    if level_ == 'debug':
        logging.basicConfig(level=logging.DEBUG, filename=fname, filemode=mode)

    elif level_ == 'release':
        logging.basicConfig(level=logging.INFO, filename=fname, filemode=mode)
    return logger, today_s


(push_i, push_fl, push_str, send_list, send_obj) = range(5)


def vm(buffer, logger=None, date=None):
    len_ = 25
    if logger:
        logger.info(logger.debug(f'Log started {date}'))
    vm_is_running = True
    ip = 0
    sp = -1
    sp_str = -1
    steck = [0] * len_
    op = buffer[ip]
    while ip < len(buffer):
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])  # Из строкового параметра
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])  # Из строкового параметра
        elif op == push_str:
            sp_str += 1
            ip += 1
            steck[sp] = buffer[ip]
        ip += 1
        op = buffer[ip]


if __name__ == '__main__':
    loger, date = get_loger("debug", "log_td.txt", "w")
    p1 = ()
    vm(p1, loger, date)
 