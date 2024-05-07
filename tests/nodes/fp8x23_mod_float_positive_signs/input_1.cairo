use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 61705264, sign: false });
    data.append(FP8x23 { mag: 11403109, sign: false });
    data.append(FP8x23 { mag: 67750408, sign: false });
    data.append(FP8x23 { mag: 26173542, sign: false });
    data.append(FP8x23 { mag: 18181258, sign: false });
    data.append(FP8x23 { mag: 82275936, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
