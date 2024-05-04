use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 55311908, sign: true });
    data.append(FP8x23 { mag: 72446096, sign: true });
    data.append(FP8x23 { mag: 22868794, sign: true });
    data.append(FP8x23 { mag: 25548366, sign: true });
    data.append(FP8x23 { mag: 39887192, sign: true });
    data.append(FP8x23 { mag: 9909354, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
