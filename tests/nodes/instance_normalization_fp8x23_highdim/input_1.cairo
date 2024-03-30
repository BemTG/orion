use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6311994, sign: true });
    data.append(FP8x23 { mag: 4510629, sign: true });
    data.append(FP8x23 { mag: 5387697, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
