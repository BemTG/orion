import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Abs(RunAll):
    @staticmethod
    def abs_i32():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
        y = abs(x)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "abs_i32"
        make_test([x], y, "input_0.abs()", name)

    @staticmethod
    def abs_i8():
        x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
        y = abs(x)

        x = Tensor(Dtype.I8, x.shape, x.flatten())
        y = Tensor(Dtype.I8, y.shape, y.flatten())

        name = "abs_i8"
        make_test([x], y, "input_0.abs()", name)

    @staticmethod
    def abs_fp8x23():
        x = to_fp(np.random.randint(-127, 127, (2, 2)
                                    ).astype(np.int64), FixedImpl.FP8x23)
        y = abs(x)

        x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
        y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

        name = "abs_fp8x23"
        make_test([x], y, "input_0.abs()", name)

    @staticmethod
    def abs_fp16x16():
        x = to_fp(np.random.randint(-127, 127, (2, 2)
                                    ).astype(np.int64), FixedImpl.FP16x16)
        y = abs(x)

        x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
        y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

        name = "abs_fp16x16"
        make_test([x], y, "input_0.abs()", name)
