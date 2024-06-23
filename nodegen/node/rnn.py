import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

from typing import Any


class RNNHelper:
    def __init__(self, **params: Any) -> None:
        # RNN Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        LAYOUT = "layout"

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[str(W)].shape[0]

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
                else np.zeros(2 * hidden_size, dtype=np.float32)
            )
            h_0 = (
                params[H_0]
                if H_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )

            self.X = x
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size])
        h_list = []

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            H = self.f(
                np.dot(x, np.transpose(self.W))
                + np.dot(H_t, np.transpose(self.R))
                + np.add(*np.split(self.B, 2))
            )
            h_list.append(H)
            H_t = H

        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]

        return Y, Y_h



class Rnn(RunAll):
    # We test here with fp16x16 implementation.    
    @staticmethod
    def rnn_fp16x16_default_params():
        X =  np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
        input_size = 2
        hidden_size = 4
        weight_scale = 0.1

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn  = RNNHelper(X=X, W=W, R=R)
        Y, Y_h = rnn.step()
        
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

        name = "rnn_fp16x16_default_params"
        func_sig = "NNTrait::rnn("
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
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W], result, func_sig, name, Trait.NN)



    @staticmethod
    def rnn_fp16x16_initial_bias(): 
        X =  np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
            np.float32
        )


        input_size = 3
        hidden_size = 5
        custom_bias = 0.1
        weight_scale = 0.1

        # W = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, input_size)).astype(np.float64), 1)
        # R = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, hidden_size)).astype(np.float64), 1)
        # # Adding custom bias
        # W_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # R_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # B = np.round(np.concatenate((W_B, R_B), axis=1),1)

        
        W = weight_scale * np.ones(
            (1,  hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, hidden_size, hidden_size)
        ).astype(np.float64)

        W_B = custom_bias * np.ones((1, hidden_size)).astype(
            np.float64
        )
        R_B = np.zeros((1,hidden_size)).astype(np.float64)
        B = np.concatenate((W_B, R_B), axis=1)

       

        
        rnn = RNNHelper(X=X, W=W, R=R, B=B)
        Y, Y_h = rnn.step()
        
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

        name = "rnn_fp16x16_initial_bias"
        func_sig = "NNTrait::rnn("
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
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B], result, func_sig, name, Trait.NN)

    @staticmethod
    def rnn_fp16x16_seq_length():
        X = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                            ]
                        ).astype(np.float32)

        
        # X =  np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(
        #     np.float32
        # )


        input_size = 3
        hidden_size = 5
        custom_bias = 0.1
        weight_scale = 0.1

        # W = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, input_size)).astype(np.float64), 1)
        # R = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, hidden_size)).astype(np.float64), 1)
        # # Adding custom bias
        # W_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # R_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # B = np.round(np.concatenate((W_B, R_B), axis=1),1)

        
        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
        R_B = np.zeros((1, hidden_size)).astype(np.float32)
        B = np.concatenate((W_B, R_B), axis=1)


        
        rnn = RNNHelper(X=X, W=W, R=R, B=B)
        Y, Y_h = rnn.step()
        
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

        name = "rnn_fp16x16_seq_length"
        func_sig = "NNTrait::rnn("
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
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W, B], result, func_sig, name, Trait.NN)


     

    @staticmethod
    def rnn_fp16x16_with_batchwise(): 
        X = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float32)

        input_size = 2
        hidden_size = 4
        weight_scale = 0.5
        layout = 1
        

        W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
        R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

        rnn = RNNHelper(X=X, W=W, R=R, layout=layout)
        Y, Y_h = rnn.step()
        
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


        name = "rnn_fp16x16_with_batchwise"
        func_sig = "NNTrait::rnn("
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
        func_sig += " Option::Some(1)," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, R, W ], result, func_sig, name, Trait.NN)

    

    rnn_fp16x16_default_params()
    rnn_fp16x16_initial_bias()
    rnn_fp16x16_with_batchwise()
    rnn_fp16x16_seq_length()
