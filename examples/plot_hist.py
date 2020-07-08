import matplotlib.pyplot as plt


def plot_history_(_file:str,history:History,name_gr:str,logger:logging.Logger)->None:
    """
    Постройка графика
    :param _file: куда сохранять
    :param history: обьект история
    :param name_gr: имя графика
    :param logger: обьект логер
    :return: None
    """
    fig:plt.Figure=None
    ax:plt.Axes=None
    fig, ax=plt.subplots()
    plt.text(0.1, 1.1, name_gr)
    ax.plot(history.history["loss"],label="Уменьшение значения целевой функции")
    plt.plot(history.history['acc'],label="Доля верных ответов на обучающем наборе")
    if 'val_acc' in history.history: # Если работаем с val_acc
        plt.plot(history.history['val_acc'],label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов / loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    logger.info("Graphic saved")
    plt.show()