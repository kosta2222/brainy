using System;
using System.Collections.Generic;


namespace Brainy
{
    public enum Ops
    {
        TRESHOLD_FUNC = 0,
        TRESHOLD_FUNC_DERIV = 1,
        SIGMOID = 2,
        SIGMOID_DERIV = 3,
        RELU = 4,
        RELU_DERIV = 5,
        TAN = 6,
        TAN_DERIV = 7,
        INIT_W_MY = 8,
        INIT_W_RANDOM = 9,
        LEAKY_RELU = 10,
        LEAKY_RELU_DERIV = 11,
        INIT_W_CONST = 12,
        INIT_RANDN = 13,
        SOFTMAX = 14,
        SOFTMAX_DERIV = 15,
        PIECE_WISE_LINEAR = 16,
        PIECE_WISE_LINEAR_DERIV = 17,
        TRESHOLD_FUNC_HALF = 18,
        TRESHOLD_FUNC_HALF_DERIV = 19,
        MODIF_MSE = 20,

        DENSE = 21

    }

    public class Dense
    {
        public int in_;
        public int out_;

        public double[,] matrix;

        public int act_func;
        public double[] hidden;
        public double[] errors;
        public bool with_bias;
        public double[] biasses;
        public double[] biasses_errors;
        public Dense()
        {  // конструктор
            in_ = 0;  // количество входов слоя
            out_ = 0;  // количество выходов слоя
            matrix = new double[10, 10];  // матрица весов
            act_func = (int)Ops.RELU;
            hidden = new double[10];  // вектор после функции активации
            errors = new double[10];  // вектор ошибок слоя
            with_bias = false;
            biasses = new double[10];
            biasses_errors = new double[10];

        }
    }
    public class NetCon
    {

        private Dense[] net_dense;
        private int sp_d;

        private bool ready;
        private List<int> b_c_forward;
        private List<int> b_c_backward_tmp;
        private int[] b_c_backward;
        int ip;
        double alpha_sigmoid;
        double alpha_tan;
        double beta_tan;
        double alpha_leaky_relu;

        public NetCon(int alpha_sigmoid_ = 1, int alpha_tan_ = 1, int beta_tan_ = 1, int alpha_leaky_relu_ = 1)
        {
            net_dense = new Dense[3];  // стек слоев
            for (int elem = 0; elem < 3; elem++)
            {
                net_dense[elem] = new Dense();
            }
            sp_d = -1;  // алокатор для слоев
            alpha_sigmoid = alpha_sigmoid_;
            alpha_tan = alpha_tan_;
            beta_tan = beta_tan_;
            alpha_leaky_relu = alpha_leaky_relu_;

            ready = false;

            b_c_forward = new List<int>();
            b_c_backward_tmp = new List<int>();
            b_c_backward = null;

            ip = 0;


        }
        public Dense[] get_net_dense()
        {
            return net_dense;
        }
        public void cr_dense(int inn = 0, int outt = 0, int act_func = 0, bool with_bias = true, int init_w = (int)Ops.INIT_W_MY)
        {
            Dense layer;
            sp_d += 1;
            layer = net_dense[sp_d];
            layer.in_ = inn;
            layer.out_ = outt;
            layer.act_func = act_func;

            if (with_bias)
                layer.with_bias = true;
            else
                layer.with_bias = false;



            for (int row = 0; row < outt; row++)
            {
                for (int elem = 0; elem < inn; elem++)
                {


                    layer.matrix[row, elem] = operations(
                    init_w, 0);


                }
                if (layer.with_bias)  // сколько рядов столько и элементов в векторе биасов
                    layer.biasses[row] = operations(
                  init_w, 0);
            }

            // просто байткод для прямого распространения - стек
            b_c_forward.Add((int)Ops.DENSE);
            b_c_forward.Add(sp_d);
            // байткод будет наоборот
            b_c_backward_tmp.Add(sp_d);
            b_c_backward_tmp.Add((int)Ops.DENSE);

        }
        public void make_hidden(Dense layer, double[] inputs)
        {
            double sum;
            double val;

            int out_ = layer.out_;
            int in_ = layer.in_;

            for (int row = 0; row < out_; row++)
            {
                sum = 0;
                for (int elem = 0; elem < in_; elem++)
                {

                    sum += layer.matrix[row, elem] * inputs[elem];

                }
                if (layer.with_bias)  // сколько рядов столько и элементов в векторе биасов
                    sum += layer.biasses[row];

                val = operations(layer.act_func, sum);
                layer.hidden[row] = val;
            }
        }

        public double[] get_hidden(Dense objLay)
        {
            return objLay.hidden;
        }

        public double[] feed_forwarding(double[] in_puts)
        {
            int op;
            int arg;
            Dense layer;
            Dense layer_prev;
            Dense last_layer;
            int len_b_c_forward;

            len_b_c_forward = b_c_forward.Count;
            // проход по байт кодам эмитирует цикл
            while (ip < len_b_c_forward)
            {
                op = b_c_forward[ip];
                if (op == (int)Ops.DENSE)
                {
                    ip += 1;
                    arg = b_c_forward[ip];

                    if (arg == 0) // это первый слой
                    {
                        layer = net_dense[0];
                        make_hidden(layer, in_puts);
                    }
                    else // остальные слои
                    {
                        layer = net_dense[arg];
                        layer_prev = net_dense[arg - 1];
                        make_hidden(layer, get_hidden(layer_prev));
                    }
                    ip += 1;
                }

            }

            ip = 0;  //сбрасываем ip так прямое распространение будет в цикле

            last_layer = net_dense[sp_d]; // последний слой

            return get_hidden(last_layer);

        }

        public void backpropagate(double[] y, double[] x, double l_r)
        {

            int op;
            int arg;
            Dense layer;
            Dense layer_next;
            Dense layer_prev;
            int len_b_c_bacward;
            try
            {
                // байт-код по которому работаем
                len_b_c_bacward = b_c_backward.Length;

                // проход по байт кодам эмитирует цикл
                // расчет ошибок
                while (ip < len_b_c_bacward)
                {

                    op = b_c_backward[ip];
                    if (op == (int)Ops.DENSE)
                    {
                        ip += 1;
                        arg = b_c_backward[ip]; // i аргумент байт-кода напр DENSE 1
                        layer = net_dense[arg];
                        if (arg == sp_d) // если это последний слой
                            calc_out_error(layer, y);
                        else // остальные слои
                        {
                            layer_next = net_dense[arg + 1];
                            calc_hid_error(layer, layer_next);
                        }

                    }
                    ip += 1;
                }


                ip = 0;
                // обновление весов
                while (ip < len_b_c_bacward)
                {
                    op = b_c_backward[ip];
                    if (op == (int)Ops.DENSE)
                    {
                        ip += 1;
                        arg = b_c_backward[ip];
                        layer = net_dense[arg];

                        if (arg == 0)  // если это первый слой
                        {
                            upd_matrix(net_dense[arg], net_dense[arg].errors,
                                            x, l_r);
                        }
                        else
                        {
                            layer_prev = net_dense[arg - 1];
                            upd_matrix(layer, layer.errors,
                                            layer_prev.hidden, l_r);
                        }
                    }
                    ip += 1;
                }

                ip = 0;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        public double[] answer_nn_direct(double[] in_puts)
        {
            return feed_forwarding(in_puts);
        }


        public void end()
        {
            int len_b_c_backward = b_c_backward_tmp.Count;
            b_c_backward = new int[len_b_c_backward];
            for (int i = 0; i < len_b_c_backward; i++)
            {
                // из очереди в байткод
                b_c_backward[i] = b_c_backward_tmp[len_b_c_backward - 1 - i];
            }
            // foreach (int item in b_c_backward)
            // {
            //     Console.WriteLine("b c bac {0}", item);
            // }

        }

        public void calc_out_error(Dense layer, double[] targets)
        {
            double tmp_v;
            int out_ = layer.out_;

            for (int row = 0; row < out_; row++)
            {
                tmp_v = (layer.hidden[row] - targets[row]) * operations(
                    layer.act_func + 1, layer.hidden[row]);
                layer.errors[row] = tmp_v;
                if (layer.with_bias)
                    layer.biasses_errors[row] = layer.errors[row];
            }
        }

        public void calc_hid_error(Dense layer, Dense layer_next)
        {
            double sum;
            int in_ = layer_next.in_;
            int out_ = layer_next.out_;

            for (int elem = 0; elem < in_; elem++)
            {
                sum = 0;
                for (int row = 0; row < out_; row++)
                {
                    sum += layer_next.matrix[row, elem] * layer_next.errors[row];
                }
                layer.errors[elem] = sum * operations(
                    layer.act_func + 1, layer.hidden[elem]);
                if (layer.with_bias)
                    layer.biasses_errors[elem] = layer.errors[elem];
            }
        }

        public void upd_matrix(Dense layer, double[] errors, double[] inputs, double lr)
        {
            double error;
            double error_bias;
            int in_;
            int out_;
            in_ = layer.in_;
            out_ = layer.out_;

            for (int row = 0; row < out_; row++)
            {
                error = errors[row];
                error_bias = layer.biasses_errors[row];
                for (int elem = 0; elem < in_; elem++)
                {
                    layer.matrix[row, elem] -= lr * error * inputs[elem];
                    if (layer.with_bias)
                        layer.biasses[elem] -= error_bias * 1;

                }
            }
        }

        public double[] calc_diff(double[] out_nn, double[] teacher_answ)
        {
            double[] diff;
            diff = new double[out_nn.Length];
            int len;
            len = teacher_answ.Length;

            for (int row = 0; row < len; row++)
                diff[row] = out_nn[row] - teacher_answ[row];
            return diff;
        }

        public double get_error(double[] diff)
        {
            double sum;
            int len;

            sum = 0;
            len = diff.Length;

            for (int row = 0; row < len; row++)
                sum += diff[row] * diff[row];
            return sum / len;
        }
        public double operations(int op, double x)
        {

            double y = 0;

            switch (op)
            {
                case (int)Ops.RELU:
                    {
                        if (x <= 0)
                            return 0;
                        else
                            return x;

                    }


                case (int)Ops.RELU_DERIV:
                    {
                        if (x <= 0)
                            return 0;
                        else
                            return 1;
                    }
                case (int)Ops.TRESHOLD_FUNC:
                    {
                        if (x > 0)
                            return 1;
                        else
                            return 0;
                    }
                case (int)Ops.TRESHOLD_FUNC_DERIV:
                    {
                        return 1;
                    }
                case (int)Ops.LEAKY_RELU:
                    {
                        if (x <= 0)
                            return alpha_leaky_relu;
                        else
                            return 1;
                    }
                case (int)Ops.LEAKY_RELU_DERIV:
                    {
                        if (x <= 0)
                            return alpha_leaky_relu;
                        else
                            return 1;
                    }
                case (int)Ops.SIGMOID:
                    {
                        y = 1 / (1 + Math.Exp(-alpha_sigmoid * x));
                        return y;
                    }

                case (int)Ops.SIGMOID_DERIV:
                    {
                        return alpha_sigmoid * x * (1 - x);
                    }
                case (int)Ops.TRESHOLD_FUNC_HALF:
                    {
                        if (x >= 0.5)
                            return 1;
                        else
                            return 0;
                    }
                case (int)Ops.TRESHOLD_FUNC_HALF_DERIV:
                    return 1;
                case (int)Ops.INIT_W_MY:
                    {
                        if (ready)
                        {
                            ready = false;
                            return -0.567141530112327;
                        }
                        ready = true;
                        return 0.567141530112327;
                    }
                case (int)Ops.INIT_W_RANDOM:
                    {
                        Random r = new Random();
                        return r.NextDouble();
                    }
                case (int)Ops.TAN:
                    {
                        y = alpha_tan * Math.Tanh(beta_tan * x);
                        return y;
                    }
                case (int)Ops.TAN_DERIV:
                    {
                        double c = beta_tan / alpha_tan;
                        return c * (alpha_tan * alpha_tan - x * x);
                        // return beta_tan / alpha_tan * (alpha_tan * alpha_tan - x * x);
                    }
                case (int)Ops.PIECE_WISE_LINEAR:
                    {
                        if (x >= 0.5)
                            return 1;
                        else if (x < 0.5 && x > -0.5)
                            return x;
                        else if (x <= -0.5)
                            return 0;
                    }
                    break;
                case (int)Ops.PIECE_WISE_LINEAR_DERIV:
                    {
                        if (x < 0.5 && x > -0.5)
                            return 1;
                        else
                            return 1;
                    }
                case (int)Ops.INIT_W_CONST:
                    return 0.567141530112327;
                    ;  // case Ops.in_IT_RANDN:
                    ;  //     return np.random.randn()
                default:
                    {
                        Console.WriteLine("operations unrecognized op");
                        return -1.0;
                    }
            }
            return 0;
        }
    }

    class Program
    {
        public static void Main(string[] args)
        {
            int epochs;
            double l_r;

            NetCon net_con;
            double[] inputs;
            double[] output;
            double gl_e;
            double e;
            int ep;
            double[,] train_inp;
            double[,] train_out;
            double[] output_nc;

            train_inp = new double[4, 2] { { 1, 1 }, { 0, 0 }, { 0, 1 }, { 1, 0 } };
            train_out = new double[4, 1] { { 0 }, { 0 }, { 1 }, { 1 } };

            inputs = new double[2];
            output = new double[1];

            epochs = 100000;
            ep = 0;
            l_r = 0.1;

            int single_array_ind;

            // Создаем слои
            net_con = new NetCon();
            net_con.cr_dense(2, 7, (int)Ops.TAN, true, (int)Ops.INIT_W_MY);
            net_con.cr_dense(7, 1, (int)Ops.SIGMOID, true, (int)Ops.INIT_W_MY);
            net_con.end();
            try
            {
                while (ep < epochs)
                {
                    gl_e = 0;

                    for (single_array_ind = 0; single_array_ind < 4; single_array_ind++)
                    {

                        for (int elem_array = 0; elem_array < 2; elem_array++)
                        {
                            inputs[elem_array] = train_inp[single_array_ind, elem_array];

                        }
                        for (int elem_array = 0; elem_array < 1; elem_array++)
                        {
                            output[elem_array] = train_out[single_array_ind, elem_array];
                        }

                        output_nc = net_con.feed_forwarding(inputs);
                        net_con.backpropagate(output, inputs, l_r);
                        e = net_con.get_error(net_con.calc_diff(output_nc, output));
                        gl_e += e;
                    }
                    Console.WriteLine("error {0}", gl_e);
                    Console.WriteLine("ep {0}", ep);
                    Console.WriteLine();

                    // errors_y.append(gl_e); ;
                    // epochs_x.append(ep);

                    if (gl_e < 0.001)
                        break;

                    ep += 1;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }
    }
}







