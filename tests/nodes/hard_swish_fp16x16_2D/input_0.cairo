use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60503, sign: true });
    data.append(FP16x16 { mag: 103155, sign: false });
    data.append(FP16x16 { mag: 68084, sign: true });
    data.append(FP16x16 { mag: 177810, sign: false });
    data.append(FP16x16 { mag: 145548, sign: true });
    data.append(FP16x16 { mag: 191606, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
