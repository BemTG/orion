use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 57276092, sign: true });
    data.append(FP8x23 { mag: 57193452, sign: true });
    data.append(FP8x23 { mag: 62223344, sign: true });
    data.append(FP8x23 { mag: 58141932, sign: true });
    data.append(FP8x23 { mag: 11810941, sign: true });
    data.append(FP8x23 { mag: 25115014, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
