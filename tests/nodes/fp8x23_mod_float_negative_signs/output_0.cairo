use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2764924, sign: true });
    data.append(FP8x23 { mag: 1927204, sign: true });
    data.append(FP8x23 { mag: 35862320, sign: true });
    data.append(FP8x23 { mag: 3404880, sign: true });
    data.append(FP8x23 { mag: 2478816, sign: true });
    data.append(FP8x23 { mag: 30133138, sign: true });
    TensorTrait::new(shape.span(), data.span())
}