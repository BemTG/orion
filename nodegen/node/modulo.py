import numpy as np
from nodegen.node import RunAll
from ..helpers import  make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def mod( a, b, fmod=None):  # type: ignore
    if fmod == 1:  # type: ignore
        return (np.fmod(a, b),)
    if a.dtype in (np.float16, np.float32, np.float64):
        return (np.nan_to_num(np.fmod(a, b)),)
    return (np.nan_to_num(np.mod(a, b)),)

class Modulo(RunAll):
    @staticmethod
    def fp8x23():
        def fp8x23_mod_float_mixed_signs():
            x = np.random.uniform(-10, 10, (6)).astype(np.float32)
            y = np.random.uniform(-10, 10, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_float_mixed_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp8x23_mod_float_positive_signs():
            x = np.random.uniform(0, 10, (6)).astype(np.float32)
            y = np.random.uniform(1, 11, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_float_positive_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp8x23_mod_float_negative_signs():
            x = np.random.uniform(-10, 0, (6)).astype(np.float32)
            y = np.random.uniform(-10, 0, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_float_negative_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)


        def fp8x23_mod_int_mixed_signs():
            x = np.random.uniform(-10, 10, (6)).astype(np.int32)
            y = np.random.uniform(-10, 10, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_int_mixed_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp8x23_mod_int_positive_signs():
            x = np.random.uniform(0, 10, (6)).astype(np.int32)
            y = np.random.uniform(1, 11, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_int_positive_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp8x23_mod_int_negative_signs():
            x = np.random.uniform(0, -10, (6)).astype(np.int32)
            y = np.random.uniform(-1, -11, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_mod_int_negative_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp8x23_int_mod_broadcast():
            x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
            y = np.array(7).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = None

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_int_mod_broadcast"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::None(()) )", name)

        def fp8x23_float_mod_broadcast():
            x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.float32)
            y = np.array(7).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP8x23, x.shape, to_fp(x, FixedImpl.FP8x23))
            _y = Tensor(Dtype.FP8x23, y.shape, to_fp(y, FixedImpl.FP8x23))
            _z = Tensor(Dtype.FP8x23, z.shape, to_fp(z, FixedImpl.FP8x23))

            name = "fp8x23_float_mod_broadcast"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

       

        fp8x23_mod_float_mixed_signs()
        fp8x23_mod_float_positive_signs()
        fp8x23_mod_float_negative_signs()
        fp8x23_mod_int_mixed_signs()
        fp8x23_mod_int_positive_signs()
        fp8x23_mod_int_negative_signs()
        fp8x23_int_mod_broadcast()
        fp8x23_float_mod_broadcast()


    def fp16x16():
        def fp16x16_mod_float_mixed_signs():
            x = np.random.uniform(-10, 10, (6)).astype(np.float32)
            y = np.random.uniform(-10, 10, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_float_mixed_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_mod_float_positive_signs():
            x = np.random.uniform(0, 10, (6)).astype(np.float32)
            y = np.random.uniform(1, 11, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_float_positive_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_mod_float_negative_signs():
            x = np.random.uniform(-10, 0, (6)).astype(np.float32)
            y = np.random.uniform(-10, 0, (6)).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_float_negative_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_mod_int_mixed_signs():
            x = np.random.uniform(-10, 10, (6)).astype(np.int32)
            y = np.random.uniform(-10, 10, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_int_mixed_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_mod_int_positive_signs():
            x = np.random.uniform(0, 10, (6)).astype(np.int32)
            y = np.random.uniform(1, 11, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_int_positive_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_mod_int_negative_signs():
            x = np.random.uniform(-10, 0, (6)).astype(np.int32)
            y = np.random.uniform(-10, 0, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = "false"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_mod_int_negative_signs"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_} ))", name)

        def fp16x16_int_mod_broadcast():
            x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
            y = np.array(7).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = None

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_int_mod_broadcast"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::None(()) )", name)

        def fp16x16_float_mod_broadcast():
            x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.float32)
            y = np.array(7).astype(np.float32)
            z = np.fmod(x, y)
            fmod_ = "true"

            _x = Tensor(Dtype.FP16x16, x.shape, to_fp(x, FixedImpl.FP16x16))
            _y = Tensor(Dtype.FP16x16, y.shape, to_fp(y, FixedImpl.FP16x16))
            _z = Tensor(Dtype.FP16x16, z.shape, to_fp(z, FixedImpl.FP16x16))

            name = "fp16x16_float_mod_broadcast"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::Some({fmod_}))", name)

       

        fp16x16_mod_float_mixed_signs()
        fp16x16_mod_float_positive_signs()
        fp16x16_mod_float_negative_signs()
        fp16x16_mod_int_mixed_signs()
        fp16x16_mod_int_positive_signs()
        fp16x16_mod_int_negative_signs()
        fp16x16_float_mod_broadcast()
        fp16x16_int_mod_broadcast()


    def uint32():
        def uint32_mod():
            x = np.random.uniform(0, 10, (6)).astype(np.int32)
            y = np.random.uniform(1, 11, (6)).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = None

            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            _z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "uint32_mod"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::None(()) )", name)

        def uint32_mod_broadcast():
            x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
            y = np.array(7).astype(np.int32)
            z = np.mod(x, y)
            fmod_ = None

            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            _z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "uint32_mod_broadcast"
            make_test([_x, _y], _z, f"input_0.modulo( @input_1 , Option::None(()) )", name)

        uint32_mod()
        uint32_mod_broadcast()



