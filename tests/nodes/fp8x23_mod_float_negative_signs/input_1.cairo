use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 46747664, sign: true });
    data.append(FP8x23 { mag: 43764052, sign: true });
    data.append(FP8x23 { mag: 49608032, sign: true });
    data.append(FP8x23 { mag: 185695, sign: true });
    data.append(FP8x23 { mag: 44340656, sign: true });
    data.append(FP8x23 { mag: 56671176, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
