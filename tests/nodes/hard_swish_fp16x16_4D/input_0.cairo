use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 83052, sign: false });
    data.append(FP16x16 { mag: 123112, sign: false });
    data.append(FP16x16 { mag: 89569, sign: true });
    data.append(FP16x16 { mag: 27590, sign: true });
    data.append(FP16x16 { mag: 130843, sign: true });
    data.append(FP16x16 { mag: 51151, sign: false });
    data.append(FP16x16 { mag: 88296, sign: true });
    data.append(FP16x16 { mag: 161391, sign: true });
    data.append(FP16x16 { mag: 136348, sign: false });
    data.append(FP16x16 { mag: 80118, sign: false });
    data.append(FP16x16 { mag: 38065, sign: false });
    data.append(FP16x16 { mag: 26180, sign: false });
    data.append(FP16x16 { mag: 2320, sign: false });
    data.append(FP16x16 { mag: 137551, sign: true });
    data.append(FP16x16 { mag: 69267, sign: true });
    data.append(FP16x16 { mag: 148523, sign: true });
    data.append(FP16x16 { mag: 39449, sign: false });
    data.append(FP16x16 { mag: 96818, sign: false });
    data.append(FP16x16 { mag: 152389, sign: false });
    data.append(FP16x16 { mag: 126678, sign: true });
    data.append(FP16x16 { mag: 126783, sign: true });
    data.append(FP16x16 { mag: 115512, sign: false });
    data.append(FP16x16 { mag: 137934, sign: false });
    data.append(FP16x16 { mag: 136119, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
