use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 36093, sign: true });
    data.append(FP16x16 { mag: 37297, sign: true });
    data.append(FP16x16 { mag: 101399, sign: true });
    data.append(FP16x16 { mag: 71296, sign: false });
    data.append(FP16x16 { mag: 27495, sign: false });
    data.append(FP16x16 { mag: 70034, sign: true });
    data.append(FP16x16 { mag: 150794, sign: true });
    data.append(FP16x16 { mag: 114644, sign: false });
    data.append(FP16x16 { mag: 8013, sign: false });
    data.append(FP16x16 { mag: 46732, sign: false });
    data.append(FP16x16 { mag: 92781, sign: true });
    data.append(FP16x16 { mag: 18290, sign: true });
    data.append(FP16x16 { mag: 18845, sign: false });
    data.append(FP16x16 { mag: 49811, sign: false });
    data.append(FP16x16 { mag: 79698, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
