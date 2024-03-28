use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4852, sign: true });
    data.append(FP16x16 { mag: 40928, sign: true });
    data.append(FP16x16 { mag: 11066, sign: false });
    data.append(FP16x16 { mag: 23555, sign: true });
    data.append(FP16x16 { mag: 102842, sign: false });
    data.append(FP16x16 { mag: 165181, sign: true });
    data.append(FP16x16 { mag: 12937, sign: true });
    data.append(FP16x16 { mag: 30810, sign: true });
    data.append(FP16x16 { mag: 18774, sign: true });
    data.append(FP16x16 { mag: 105218, sign: false });
    data.append(FP16x16 { mag: 4076, sign: true });
    data.append(FP16x16 { mag: 108560, sign: true });
    data.append(FP16x16 { mag: 19907, sign: false });
    data.append(FP16x16 { mag: 1443, sign: true });
    data.append(FP16x16 { mag: 57349, sign: true });
    data.append(FP16x16 { mag: 11611, sign: true });
    data.append(FP16x16 { mag: 4640, sign: true });
    data.append(FP16x16 { mag: 1012, sign: false });
    data.append(FP16x16 { mag: 35937, sign: false });
    data.append(FP16x16 { mag: 35129, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
