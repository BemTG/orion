use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 63251408, sign: false });
    data.append(FP8x23 { mag: 39794780, sign: false });
    data.append(FP8x23 { mag: 80203768, sign: false });
    data.append(FP8x23 { mag: 64693900, sign: true });
    data.append(FP8x23 { mag: 68532128, sign: false });
    data.append(FP8x23 { mag: 77490280, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
