import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

from typing import Any


class LSTMHelper:
    def __init__(self, **params: Any) -> None:
        # LSTM Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        C_0 = "initial_c"
        P = "P"
        LAYOUT = "layout"
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params.get(LAYOUT, 0)
            x = params[X]
            x = x if layout == 0 else np.swapaxes(x, 0, 1)
            b = (
                params[B]
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float32)
            )
            p = (
                params[P]
                if P in params
                else np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            )
            h_0 = (
                params[H_0]
                if H_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )
            c_0 = (
                params[C_0]
                if C_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]
        
        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        [p_i, p_o, p_f] = np.split(self.P, 3)
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = (
                np.dot(x, np.transpose(self.W))
                + np.dot(H_t, np.transpose(self.R))
                + np.add(*np.split(self.B, 2))
            )
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h



class Lstm(RunAll):
    # We test here with fp16x16 implementation.    
    @staticmethod
    def fp16x16_default_params():
        X =  np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
        input_size = 2
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        lstm = LSTMHelper(X=X, W=W, R=R)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP16x16, X.shape, to_fp(
            X.flatten(), FixedImpl.FP16x16))
        W = Tensor(Dtype.FP16x16, W.shape, to_fp(
            W.flatten(), FixedImpl.FP16x16))
        R = Tensor(Dtype.FP16x16, R.shape, to_fp(
            R.flatten(), FixedImpl.FP16x16))
        
        result = [
                Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16)),
                Tensor(Dtype.FP16x16, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP16x16))
                ]

        name = "lstm_fp16x16_default_params"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W], result, func_sig, name, Trait.NN)



    @staticmethod
    def fp16x16_initial_bias(): 
        X =  np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )


        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 4

        # W = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, input_size)).astype(np.float64), 1)
        # R = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, hidden_size)).astype(np.float64), 1)
        # # Adding custom bias
        # W_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # R_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # B = np.round(np.concatenate((W_B, R_B), axis=1),1)

        
        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
            np.float64
        )
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float64)
        B = np.concatenate((W_B, R_B), axis=1)

       

        
        gru = LSTMHelper(X=X, W=W, R=R, B=B)
        Y, Y_h = gru.step()
        
        X = Tensor(Dtype.FP16x16, X.shape, to_fp(
            X.flatten(), FixedImpl.FP16x16))
        W = Tensor(Dtype.FP16x16, W.shape, to_fp(
            W.flatten(), FixedImpl.FP16x16))
        R = Tensor(Dtype.FP16x16, R.shape, to_fp(
            R.flatten(), FixedImpl.FP16x16))
        B = Tensor(Dtype.FP16x16, B.shape, to_fp(
            B.flatten(), FixedImpl.FP16x16))
        
        result = [
                Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16)),
                Tensor(Dtype.FP16x16, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP16x16))
                ]

        name = "lstm_fp16x16_initial_bias"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::Some(input_3)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B], result, func_sig, name, Trait.NN)

    @staticmethod
    def fp16x16_batchwise():
        X =  np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)
        
        input_size = 2
        hidden_size = 7
        weight_scale = 0.3
        number_of_gates = 4
        layout = 1

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        lstm = LSTMHelper(X=X, W=W, R=R, layout=layout)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP16x16, X.shape, to_fp(
            X.flatten(), FixedImpl.FP16x16))
        W = Tensor(Dtype.FP16x16, W.shape, to_fp(
            W.flatten(), FixedImpl.FP16x16))
        R = Tensor(Dtype.FP16x16, R.shape, to_fp(
            R.flatten(), FixedImpl.FP16x16))
        
        result = [
                Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16)),
                Tensor(Dtype.FP16x16, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP16x16))
                ]

        name = "lstm_fp16x16_batchwise"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(1)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W], result, func_sig, name, Trait.NN)


     

    @staticmethod
    def fp16x16_with_peepholes(): 
        X = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]).astype(
            np.float32
        )

        input_size = 4
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4
        number_of_peepholes = 3
        

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
        # seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
        init_h = np.zeros((1, X.shape[1], hidden_size)).astype(np.float32)
        init_c = np.zeros((1, X.shape[1], hidden_size)).astype(np.float32)
        P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(
            np.float32
        )


        lstm = LSTMHelper(X=X, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP16x16, X.shape, to_fp(
            X.flatten(), FixedImpl.FP16x16))
        W = Tensor(Dtype.FP16x16, W.shape, to_fp(
            W.flatten(), FixedImpl.FP16x16))
        R = Tensor(Dtype.FP16x16, R.shape, to_fp(
            R.flatten(), FixedImpl.FP16x16))
        B = Tensor(Dtype.FP16x16, B.shape, to_fp(
            B.flatten(), FixedImpl.FP16x16))
        P = Tensor(Dtype.FP16x16, P.shape, to_fp(
            P.flatten(), FixedImpl.FP16x16))
        
        init_c = Tensor(Dtype.FP16x16, init_c.shape, to_fp(
            init_c.flatten(), FixedImpl.FP16x16))
        
        init_h = Tensor(Dtype.FP16x16, init_h.shape, to_fp(
            init_h.flatten(), FixedImpl.FP16x16))
        
        
        result = [
                Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16)),
                Tensor(Dtype.FP16x16, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP16x16))
                ]


        name = "lstm_fp16x16_with_peepholes"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::Some(input_3)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(input_4)," 
        func_sig += " Option::Some(input_5)," 
        func_sig += " Option::Some(input_6)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B, init_h, init_c, P ], result, func_sig, name, Trait.NN)

    

    fp16x16_default_params()
    fp16x16_initial_bias()
    fp16x16_batchwise()
    fp16x16_with_peepholes()


# ---------------------------------------------------------------------------------------------
        
    @staticmethod
    def fp8x23_default_params():
        X =  np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
        input_size = 2
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        lstm = LSTMHelper(X=X, W=W, R=R)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP8x23, X.shape, to_fp(
            X.flatten(), FixedImpl.FP8x23))
        W = Tensor(Dtype.FP8x23, W.shape, to_fp(
            W.flatten(), FixedImpl.FP8x23))
        R = Tensor(Dtype.FP8x23, R.shape, to_fp(
            R.flatten(), FixedImpl.FP8x23))
        
        result = [
                Tensor(Dtype.FP8x23, Y.shape, to_fp(Y.flatten(), FixedImpl.FP8x23)),
                Tensor(Dtype.FP8x23, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP8x23))
                ]

        name = "lstm_FP8x23_default_params"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W], result, func_sig, name, Trait.NN)



    @staticmethod
    def FP8x23_initial_bias(): 
        X =  np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )


        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 4

        # W = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, input_size)).astype(np.float64), 1)
        # R = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, hidden_size)).astype(np.float64), 1)
        # # Adding custom bias
        # W_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # R_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # B = np.round(np.concatenate((W_B, R_B), axis=1),1)

        
        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
            np.float64
        )
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float64)
        B = np.concatenate((W_B, R_B), axis=1)

       

        
        gru = LSTMHelper(X=X, W=W, R=R, B=B)
        Y, Y_h = gru.step()
        
        X = Tensor(Dtype.FP8x23, X.shape, to_fp(
            X.flatten(), FixedImpl.FP8x23))
        W = Tensor(Dtype.FP8x23, W.shape, to_fp(
            W.flatten(), FixedImpl.FP8x23))
        R = Tensor(Dtype.FP8x23, R.shape, to_fp(
            R.flatten(), FixedImpl.FP8x23))
        B = Tensor(Dtype.FP8x23, B.shape, to_fp(
            B.flatten(), FixedImpl.FP8x23))
        
        result = [
                Tensor(Dtype.FP8x23, Y.shape, to_fp(Y.flatten(), FixedImpl.FP8x23)),
                Tensor(Dtype.FP8x23, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP8x23))
                ]

        name = "lstm_FP8x23_initial_bias"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::Some(input_3)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B], result, func_sig, name, Trait.NN)

    @staticmethod
    def FP8x23_batchwise():
        X =  np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)
        
        input_size = 2
        hidden_size = 7
        weight_scale = 0.3
        number_of_gates = 4
        layout = 1

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        lstm = LSTMHelper(X=X, W=W, R=R, layout=layout)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP8x23, X.shape, to_fp(
            X.flatten(), FixedImpl.FP8x23))
        W = Tensor(Dtype.FP8x23, W.shape, to_fp(
            W.flatten(), FixedImpl.FP8x23))
        R = Tensor(Dtype.FP8x23, R.shape, to_fp(
            R.flatten(), FixedImpl.FP8x23))
        
        result = [
                Tensor(Dtype.FP8x23, Y.shape, to_fp(Y.flatten(), FixedImpl.FP8x23)),
                Tensor(Dtype.FP8x23, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP8x23))
                ]

        name = "lstm_FP8x23_batchwise"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(1)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W], result, func_sig, name, Trait.NN)

     

    @staticmethod
    def FP8x23_with_peepholes(): 
        X = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]).astype(
            np.float32
        )

        input_size = 4
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4
        number_of_peepholes = 3
        

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
        # seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
        init_h = np.zeros((1, X.shape[1], hidden_size)).astype(np.float32)
        init_c = np.zeros((1, X.shape[1], hidden_size)).astype(np.float32)
        P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(
            np.float32
        )


        lstm = LSTMHelper(X=X, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
        Y, Y_h = lstm.step()
        
        X = Tensor(Dtype.FP8x23, X.shape, to_fp(
            X.flatten(), FixedImpl.FP8x23))
        W = Tensor(Dtype.FP8x23, W.shape, to_fp(
            W.flatten(), FixedImpl.FP8x23))
        R = Tensor(Dtype.FP8x23, R.shape, to_fp(
            R.flatten(), FixedImpl.FP8x23))
        B = Tensor(Dtype.FP8x23, B.shape, to_fp(
            B.flatten(), FixedImpl.FP8x23))
        P = Tensor(Dtype.FP8x23, P.shape, to_fp(
            P.flatten(), FixedImpl.FP8x23))
        
        init_c = Tensor(Dtype.FP8x23, init_c.shape, to_fp(
            init_c.flatten(), FixedImpl.FP8x23))
        
        init_h = Tensor(Dtype.FP8x23, init_h.shape, to_fp(
            init_h.flatten(), FixedImpl.FP8x23))
        
        
        result = [
                Tensor(Dtype.FP8x23, Y.shape, to_fp(Y.flatten(), FixedImpl.FP8x23)),
                Tensor(Dtype.FP8x23, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP8x23))
                ]


        name = "lstm_FP8x23_with_peepholes"
        func_sig = "NNTrait::lstm("
        func_sig += " @input_0,"
        func_sig += " @input_1,"
        func_sig += " @input_2,"
        func_sig += " Option::Some(input_3)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(input_4)," 
        func_sig += " Option::Some(input_5)," 
        func_sig += " Option::Some(input_6)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B, init_h, init_c, P ], result, func_sig, name, Trait.NN)


    fp8x23_default_params()
    FP8x23_initial_bias()
    FP8x23_batchwise()
    FP8x23_with_peepholes()



    
    