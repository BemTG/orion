use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14502777, sign: false });
    data.append(FP8x23 { mag: 1129999, sign: true });
    data.append(FP8x23 { mag: 2917025, sign: false });
    data.append(FP8x23 { mag: 3506855, sign: false });
    data.append(FP8x23 { mag: 8218504, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
