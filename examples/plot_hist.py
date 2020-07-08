import matplotlib.pyplot as plt


def plot_history_(_file:str,history:History,name_gr:str,logger:logging.Logger)->None:
    """
    ��������� �������
    :param _file: ���� ���������
    :param history: ������ �������
    :param name_gr: ��� �������
    :param logger: ������ �����
    :return: None
    """
    fig:plt.Figure=None
    ax:plt.Axes=None
    fig, ax=plt.subplots()
    plt.text(0.1, 1.1, name_gr)
    ax.plot(history.history["loss"],label="���������� �������� ������� �������")
    plt.plot(history.history['acc'],label="���� ������ ������� �� ��������� ������")
    if 'val_acc' in history.history: # ���� �������� � val_acc
        plt.plot(history.history['val_acc'],label='���� ������ ������� �� ����������� ������')
    plt.xlabel('����� ��������')
    plt.ylabel('���� ������ ������� / loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    logger.info("Graphic saved")
    plt.show()