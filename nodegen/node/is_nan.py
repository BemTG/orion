import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


# NaN is represented with -0
NaN = -0

class Is_nan(RunAll):

    @staticmethod
    def is_nan_fp8x23():
        def default():
            input_0 = np.array([-1.2, 0, NaN, 2.8, NaN, NaN], dtype=np.float64)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.FP8x23, input_0.shape, to_fp(
                input_0.flatten(), FixedImpl.FP8x23))
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_nan_fp8x23"
            make_test([input_0], output, "TensorTrait::is_nan(@input_0)", name)

        default()

    @staticmethod
    def is_nan_fp16x16():
        def default():
            input_0 = np.array([-1.2, 0, NaN, 2.8, NaN, NaN], dtype=np.float64)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.FP16x16, input_0.shape, to_fp(
                input_0.flatten(), FixedImpl.FP16x16))
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_nan_fp16x16"
            make_test([input_0], output, "TensorTrait::is_nan(@input_0)", name)

        default()
