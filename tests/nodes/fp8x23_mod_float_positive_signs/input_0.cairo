use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 46442456, sign: false });
    data.append(FP8x23 { mag: 40705340, sign: false });
    data.append(FP8x23 { mag: 54742288, sign: false });
    data.append(FP8x23 { mag: 44871880, sign: false });
    data.append(FP8x23 { mag: 48112428, sign: false });
    data.append(FP8x23 { mag: 45530660, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
