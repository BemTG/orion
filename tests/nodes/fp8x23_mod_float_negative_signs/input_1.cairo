use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 24374096, sign: true });
    data.append(FP8x23 { mag: 9369027, sign: true });
    data.append(FP8x23 { mag: 20921350, sign: true });
    data.append(FP8x23 { mag: 27530086, sign: true });
    data.append(FP8x23 { mag: 56993952, sign: true });
    data.append(FP8x23 { mag: 57121656, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
