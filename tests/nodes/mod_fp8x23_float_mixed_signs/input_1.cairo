use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 18246666, sign: false });
    data.append(FP8x23 { mag: 30923230, sign: false });
    data.append(FP8x23 { mag: 47011744, sign: false });
    data.append(FP8x23 { mag: 44953716, sign: false });
    data.append(FP8x23 { mag: 61887632, sign: false });
    data.append(FP8x23 { mag: 18623904, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
