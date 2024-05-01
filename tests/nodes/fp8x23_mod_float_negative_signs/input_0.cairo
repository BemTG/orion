use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 37885004, sign: true });
    data.append(FP8x23 { mag: 7442204, sign: true });
    data.append(FP8x23 { mag: 83653680, sign: true });
    data.append(FP8x23 { mag: 7379493, sign: true });
    data.append(FP8x23 { mag: 46057080, sign: true });
    data.append(FP8x23 { mag: 1475950, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
