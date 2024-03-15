use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 14267, sign: false });
    data.append(FP16x16 { mag: 39628, sign: false });
    data.append(FP16x16 { mag: 96330, sign: true });
    data.append(FP16x16 { mag: 139449, sign: true });
    data.append(FP16x16 { mag: 55769, sign: true });
    data.append(FP16x16 { mag: 1637, sign: true });
    data.append(FP16x16 { mag: 24733, sign: true });
    data.append(FP16x16 { mag: 58622, sign: false });
    data.append(FP16x16 { mag: 27298, sign: false });
    data.append(FP16x16 { mag: 12923, sign: false });
    data.append(FP16x16 { mag: 46667, sign: true });
    data.append(FP16x16 { mag: 169071, sign: true });
    data.append(FP16x16 { mag: 51220, sign: true });
    data.append(FP16x16 { mag: 44187, sign: true });
    data.append(FP16x16 { mag: 14718, sign: false });
    data.append(FP16x16 { mag: 76739, sign: false });
    data.append(FP16x16 { mag: 49947, sign: false });
    data.append(FP16x16 { mag: 15455, sign: true });
    data.append(FP16x16 { mag: 62789, sign: true });
    data.append(FP16x16 { mag: 144549, sign: true });
    data.append(FP16x16 { mag: 56096, sign: true });
    data.append(FP16x16 { mag: 16865, sign: false });
    data.append(FP16x16 { mag: 24779, sign: false });
    data.append(FP16x16 { mag: 18426, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
