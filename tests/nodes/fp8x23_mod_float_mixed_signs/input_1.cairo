use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 72947016, sign: true });
    data.append(FP8x23 { mag: 32117114, sign: false });
    data.append(FP8x23 { mag: 39309260, sign: false });
    data.append(FP8x23 { mag: 53385724, sign: false });
    data.append(FP8x23 { mag: 65803312, sign: false });
    data.append(FP8x23 { mag: 47990360, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
