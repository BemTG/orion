use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 45454404, sign: true });
    data.append(FP8x23 { mag: 1927204, sign: true });
    data.append(FP8x23 { mag: 35862320, sign: true });
    data.append(FP8x23 { mag: 75872936, sign: true });
    data.append(FP8x23 { mag: 72765760, sign: true });
    data.append(FP8x23 { mag: 30133138, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
