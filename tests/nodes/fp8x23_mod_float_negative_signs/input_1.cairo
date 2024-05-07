use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2218854, sign: true });
    data.append(FP8x23 { mag: 15392578, sign: true });
    data.append(FP8x23 { mag: 50022792, sign: true });
    data.append(FP8x23 { mag: 76633528, sign: true });
    data.append(FP8x23 { mag: 12380776, sign: true });
    data.append(FP8x23 { mag: 82535064, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
