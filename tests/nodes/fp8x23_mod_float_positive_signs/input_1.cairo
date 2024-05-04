use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 34713220, sign: false });
    data.append(FP8x23 { mag: 74308272, sign: false });
    data.append(FP8x23 { mag: 89385000, sign: false });
    data.append(FP8x23 { mag: 37987020, sign: false });
    data.append(FP8x23 { mag: 60274020, sign: false });
    data.append(FP8x23 { mag: 67582096, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
