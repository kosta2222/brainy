from .learn import answer_nn_direct
from .NN_params import NN_params

def evaluate(nn_params:NN_params, X_test: list, Y_test: list):
  try:
    scores = []
    out_nn=None
    res_acc = 0
    rows = len(X_test)
    wi_y_test = len(Y_test[0])
    elem_of_out_nn = 0
    elem_answer = 0
    is_vecs_are_equal = False
    for row in range(rows):
        x_test = X_test[row]
        y_test = Y_test[row]
        out_nn=answer_nn_direct(nn_params, x_test, 1)
        for elem in range(wi_y_test):
            elem_of_out_nn = out_nn[elem]
            elem_answer = y_test[elem]
            if (elem_of_out_nn > 0.5):
                elem_of_out_nn = 1
                # print("output vector elem -> ( %f ) " % 1, end=' ')
            else:
                elem_of_out_nn = 0
                # print("output vector elem -> ( %f ) " % 0, end=' ');
            # print("expected vector elem -> ( %f )" % elem_answer);
            if elem_of_out_nn == elem_answer:
                is_vecs_are_equal = True
            else:
                is_vecs_are_equal = False
                break
        if is_vecs_are_equal:
           # print("-Vecs are equal-")
           scores.append(1)
        else:
            # print("-Vecs are not equal-")
            scores.append(0)
    # print("in eval scores",scores)
    res_acc = sum(scores) / rows * 100
    # print("Acсuracy:%f%s"%(res_acc,"%"))
  except Exception:
      print("in evaluate")
      print("row",row)

  return res_acc

