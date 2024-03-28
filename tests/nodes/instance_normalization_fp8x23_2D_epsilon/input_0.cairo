use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2515591, sign: false });
    data.append(FP8x23 { mag: 9580341, sign: true });
    data.append(FP8x23 { mag: 810277, sign: false });
    data.append(FP8x23 { mag: 10864025, sign: false });
    data.append(FP8x23 { mag: 6870120, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
