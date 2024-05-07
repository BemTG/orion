use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 414525, sign: true });
    data.append(FP16x16 { mag: 127876, sign: true });
    data.append(FP16x16 { mag: 106421, sign: true });
    data.append(FP16x16 { mag: 301112, sign: true });
    data.append(FP16x16 { mag: 71026, sign: true });
    data.append(FP16x16 { mag: 554907, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
