from nn_constants import max_rows_orOut
from lear_func import answer_nn_direct
from NN_params import NnParams
k = 0
def evaluate(nn_params:NnParams, X_test: list, Y_test: list):
    """
    Производит (кросс-валидацию) предсказание и сверка ответов
    по всему  Обучающий набор/тестовая выборка
    :param X_test: 2D Обучающий набор/тестовая выборка
    :param Y_test: 2D Набор ответов/тестовая выборка
    :return: аккуратность в процентах
    """
    # scores=[0] * max_rows_orOut
    scores = []
    res = 0
    out_nn=None
    res_acc = 0
    rows = len(X_test)
    wi_y_test = len(Y_test[0])
    n = 0
    elem_of_out_nn = 0
    elem_answer = 0
    is_vecs_are_equal = False
    for row in range(rows):
        x_test = X_test[row]
        y_test = Y_test[row]
        # print("in cross val x_test",x_test)
        out_nn=answer_nn_direct(nn_params, x_test, 1)
        # print("in cross val out_nn",out_nn)
        # res=check_oneHotVecs(scores, out_nn, y_test, len(y_test))
        for elem in range(wi_y_test):
            elem_of_out_nn = out_nn[elem]
            elem_answer = y_test[elem]
            if (elem_of_out_nn > 0.5):
                elem_of_out_nn = 1
                print("output vector elem -> ( %f ) " % 1, end=' ')
            else:
                elem_of_out_nn = 0
                print("output vector elem -> ( %f ) " % 0, end=' ');
            print("expected vector elem -> ( %f )" % elem_answer);
            if elem_of_out_nn == elem_answer:
                is_vecs_are_equal = True
            else:
                is_vecs_are_equal = False
                break
        if is_vecs_are_equal:
           print("-Vecs are equal-")
           scores.append(1)
        else:
            print("-Vecs are not equal-")
            scores.append(0)
            # elem_of_out_nn = 0
    print("in eval scores",scores)
    res_acc = sum(scores) / rows * 100
    print("Acсuracy:%f%s"%(res_acc,"%"))
    return res_acc
# def check_oneHotVecs(scores:list, out_nn:list, y_test, len_)->int:
#     tmp_elemOf_outNN_asHot = 0
#     global k
#     for col in range(len_):
#         tmp_elemOf_outNN_asHot=out_nn[col]
#         if (tmp_elemOf_outNN_asHot > 0 ) and (tmp_elemOf_outNN_asHot > 0.5 or tmp_elemOf_outNN_asHot == 1):
#             tmp_elemOf_outNN_asHot = 1
#         else:
#             tmp_elemOf_outNN_asHot = 0
#         if (tmp_elemOf_outNN_asHot == int(y_test[col])):
#             scores[k] = 1
#             k += 1
#         else:
#             break
# def calc_accur(scores:list, rows)->float:
#     accuracy=0
#     sum=0
#     for col in range(rows):
#         sum+=scores[col]
#     accuracy=sum / rows * 100
#     return accuracy
