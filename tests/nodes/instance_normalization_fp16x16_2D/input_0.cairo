use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38967, sign: true });
    data.append(FP16x16 { mag: 58538, sign: true });
    data.append(FP16x16 { mag: 80711, sign: true });
    data.append(FP16x16 { mag: 20981, sign: true });
    data.append(FP16x16 { mag: 32667, sign: false });
    data.append(FP16x16 { mag: 8783, sign: true });
    data.append(FP16x16 { mag: 106479, sign: true });
    data.append(FP16x16 { mag: 56810, sign: true });
    data.append(FP16x16 { mag: 25009, sign: false });
    data.append(FP16x16 { mag: 62816, sign: false });
    data.append(FP16x16 { mag: 10919, sign: false });
    data.append(FP16x16 { mag: 69855, sign: false });
    data.append(FP16x16 { mag: 29208, sign: false });
    data.append(FP16x16 { mag: 7542, sign: false });
    data.append(FP16x16 { mag: 4152, sign: true });
    data.append(FP16x16 { mag: 1856, sign: false });
    data.append(FP16x16 { mag: 20106, sign: false });
    data.append(FP16x16 { mag: 65973, sign: true });
    data.append(FP16x16 { mag: 97604, sign: false });
    data.append(FP16x16 { mag: 65765, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
