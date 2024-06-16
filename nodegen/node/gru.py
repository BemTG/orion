import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

from typing import Any
import numpy as np
from typing import Any

import numpy as np
from typing import Any

import numpy as np
from typing import Any

class GRUHelper:
    def __init__(self, **params: Any) -> None:
        # GRU Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        LBR = "linear_before_reset"
        LAYOUT = "layout"
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0).astype(np.float64)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params.get(LAYOUT, 0)
            x = params[X].astype(np.float64)
            x = x if layout == 0 else np.swapaxes(x, 0, 1).astype(np.float64)
            b = (
                params[B].astype(np.float64)
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float64)
            )
            h_0 = params[H_0].astype(np.float64) if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float64)
            lbr = params.get(LBR, 0)

            self.X = x
            self.W = params[W].astype(np.float64)
            self.R = params[R].astype(np.float64)
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-x))).astype(np.float64)

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x).astype(np.float64)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size], dtype=np.float64)
        h_list = []

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r))).astype(np.float64)
        gates_r = np.transpose(np.concatenate((r_z, r_r))).astype(np.float64)
        gates_b = np.add(np.concatenate((w_bz, w_br)).astype(np.float64), np.concatenate((r_bz, r_br)).astype(np.float64))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            x = x.astype(np.float64)
            H_t = H_t.astype(np.float64)
            gates = (np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b).astype(np.float64)
            z, r = np.split(gates.astype(np.float64), 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                (np.dot(x, np.transpose(w_h).astype(np.float64))
                + np.dot(r * H_t, np.transpose(r_h).astype(np.float64))
                + w_bh + r_bh).astype(np.float64)
            )
            h_linear = self.g(
                (np.dot(x, np.transpose(w_h).astype(np.float64))
                + r * (np.dot(H_t, np.transpose(r_h).astype(np.float64)) + r_bh)
                + w_bh).astype(np.float64)
            )
            h = h_linear if self.LBR else h_default
            H = ((1 - z) * h + z * H_t).astype(np.float64)
            h_list.append(H.astype(np.float64))
            H_t = H.astype(np.float64)

        concatenated = np.concatenate(h_list).astype(np.float64)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3]).astype(np.float64)
            Y_h = Y[:, :, -1, :].astype(np.float64)

        return Y.astype(np.float32), Y_h.astype(np.float32)

    def __init__(self, **params: Any) -> None:
        # GRU Input Names
        X = "X"
        W = "W"
        R = "R"
        B = "B"
        H_0 = "initial_h"
        LBR = "linear_before_reset"
        LAYOUT = "layout"
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params:
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0).astype(np.float64)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            layout = params.get(LAYOUT, 0)
            x = params[X].astype(np.float64)
            x = x if layout == 0 else np.swapaxes(x, 0, 1).astype(np.float64)
            b = (
                params[B].astype(np.float64)
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float64)
            )
            h_0 = params[H_0].astype(np.float64) if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float64)
            lbr = params.get(LBR, 0)

            self.X = x
            self.W = params[W].astype(np.float64)
            self.R = params[R].astype(np.float64)
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr
            self.LAYOUT = layout

        else:
            raise NotImplementedError()

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        seq_length = self.X.shape[0]
        hidden_size = self.H_0.shape[-1]
        batch_size = self.X.shape[1]

        Y = np.empty([seq_length, self.num_directions, batch_size, hidden_size], dtype=np.float64)
        h_list = []

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r))).astype(np.float64)
        gates_r = np.transpose(np.concatenate((r_z, r_r))).astype(np.float64)
        gates_b = np.add(np.concatenate((w_bz, w_br)).astype(np.float64), np.concatenate((r_bz, r_br)).astype(np.float64))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            x = x.astype(np.float64)
            H_t = H_t.astype(np.float64)
            gates = (np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b).astype(np.float64)
            z, r = np.split(gates.astype(np.float64), 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                (np.dot(x, np.transpose(w_h).astype(np.float64))
                + np.dot(r * H_t, np.transpose(r_h).astype(np.float64))
                + w_bh + r_bh).astype(np.float64)
            )
            h_linear = self.g(
                (np.dot(x, np.transpose(w_h).astype(np.float64))
                + r * (np.dot(H_t, np.transpose(r_h).astype(np.float64)) + r_bh)
                + w_bh).astype(np.float64)
            )
            h = h_linear if self.LBR else h_default
            H = ((1 - z) * h + z * H_t).astype(np.float64)
            h_list.append(H.astype(np.float64))
            H_t = H.astype(np.float64)

        concatenated = np.concatenate(h_list).astype(np.float64)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3]).astype(np.float64)
            Y_h = Y[:, :, -1, :].astype(np.float64)

        return Y.astype(np.float32), Y_h.astype(np.float32)


class Gru(RunAll):
    # We test here with fp16x16 implementation.    
    @staticmethod
    def fp16x16_default_params():
        X =  np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float64)
        input_size = 2
        hidden_size = 5
        weight_scale = 0.1
        number_of_gates = 3

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        gru = GRUHelper(X=X, W=W, R=R)
        Y, Y_h = gru.step()
        
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

        name = "gru_fp16x16_default_params"
        func_sig = "NNTrait::gru("
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

        make_test([X, W, R], result, func_sig, name, Trait.NN)


    @staticmethod
    def fp16x16_default_with_initial_bias(): 
        X =  np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(np.float64)
        input_size = 3
        hidden_size = 3
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 3

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
            np.float64
        )
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float64)
        B = np.concatenate((W_B, R_B), axis=1)

        gru = GRUHelper(X=X, W=W, R=R, B=B)
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

        name = "gru_fp16x16_with_initial_bias"
        func_sig = "NNTrait::gru("
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

        make_test([X, W, R, B], result, func_sig, name, Trait.NN)


    @staticmethod
    def fp16x16_varying_sequence_length(): 
        X =  np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ]
        ).astype(np.float64)
        
        input_size = 3
        hidden_size = 5
        number_of_gates = 3


        # W = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, input_size)).astype(np.float64), 1)
        # R = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size, hidden_size)).astype(np.float64), 1)
        # # Adding custom bias
        # W_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # R_B = np.round(np.random.uniform(0.05, 0.1, (1, number_of_gates * hidden_size)).astype(np.float64),1)
        # B = np.round(np.concatenate((W_B, R_B), axis=1),1)

        weight_scale = 0.1
        custom_bias = 0.1
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

       

        
        gru = GRUHelper(X=X, W=W, R=R, B=B)
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

        name = "gru_fp16x16_varying_sequence_length" 
        func_sig = "NNTrait::gru("
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

        make_test([X, W, R, B], result, func_sig, name, Trait.NN)


    @staticmethod
    def fp16x16_with_batchwise_processing(): 
        X =  np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]).astype(np.float64)

        input_size = 2
        hidden_size = 6
        number_of_gates = 3
        weight_scale = 0.2
        layout = 1
        # custom_bias = 0.1

        

        W = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, input_size)
        ).astype(np.float64)
        R = weight_scale * np.ones(
            (1, number_of_gates * hidden_size, hidden_size)
        ).astype(np.float64)

        # # Adding custom bias
        # W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(
        #     np.float64
        # )
        # R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float64)
        # B = np.concatenate((W_B, R_B), axis=1)


        gru = GRUHelper(X=X, W=W, R=R, layout=layout)
        Y, Y_h = gru.step()
        
        X = Tensor(Dtype.FP16x16, X.shape, to_fp(
            X.flatten(), FixedImpl.FP16x16))
        W = Tensor(Dtype.FP16x16, W.shape, to_fp(
            W.flatten(), FixedImpl.FP16x16))
        R = Tensor(Dtype.FP16x16, R.shape, to_fp(
            R.flatten(), FixedImpl.FP16x16))
        # B = Tensor(Dtype.FP16x16, B.shape, to_fp(
        #     B.flatten(), FixedImpl.FP16x16))
        
        result = [
                Tensor(Dtype.FP16x16, Y.shape, to_fp(Y.flatten(), FixedImpl.FP16x16)),
                Tensor(Dtype.FP16x16, Y_h.shape, to_fp(Y_h.flatten(), FixedImpl.FP16x16))
                ]


        name = "gru_fp16x16_with_batchwise_processing" 
        func_sig = "NNTrait::gru("
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
        func_sig += f" Option::Some({layout})," 
        func_sig += " Option::None(())," 
        func_sig += " Option::Some(2) " 
        func_sig += " ) " 

        make_test([X, W, R], result, func_sig, name, Trait.NN)


