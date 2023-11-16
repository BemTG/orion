import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Div(RunAll):
    @staticmethod
    def div_u32():
        def default():
            x = np.random.randint(3, 6, (3, 3, 3)).astype(np.uint32)
            y = np.random.randint(1, 3, (3, 3, 3)).astype(np.uint32)
            z = x / y

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "div_u32"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        def broadcast():
            x = np.random.randint(3, 6, (3, 3, 3)).astype(np.uint32)
            y = np.random.randint(1, 3, (1, 3, 1)).astype(np.uint32)
            z = x / y

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "div_u32_broadcast"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        default()
        broadcast()

    @staticmethod
    def div_i32():
        def default():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.int32)
            y = np.random.randint(1, 3, (3, 3, 3)).astype(np.int32)
            z = x / y

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "div_i32"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        def broadcast():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.int32)
            y = np.random.randint(1, 3, (1, 3, 1)).astype(np.int32)
            z = x / y

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "div_i32_broadcast"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        default()
        broadcast()

    @staticmethod
    def div_i8():
        def default():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.int8)
            y = np.random.randint(1, 3, (3, 3, 3)).astype(np.int8)
            z = x / y

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "div_i8"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        def broadcast():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.int8)
            y = np.random.randint(1, 3, (1, 3, 1)).astype(np.int8)
            z = x / y

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "div_i8_broadcast"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        default()
        broadcast()

    @staticmethod
    def div_fp8x23():
        def default():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            z = x / y

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape, to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "div_fp8x23"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        def broadcast():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(1, 3, (1, 3, 1)).astype(np.float64)
            z = x / y

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape, to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "div_fp8x23_broadcast"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        default()
        broadcast()

    @staticmethod
    def div_fp16x16():
        def default():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            z = x / y

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape, to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "div_fp16x16"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        def broadcast():
            x = np.random.randint(1, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(1, 3, (1, 3, 1)).astype(np.float64)
            z = x / y

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape, to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "div_fp16x16_broadcast"
            make_test([x, y], z, "input_0.div(@input_1)", name)

        default()
        broadcast()
