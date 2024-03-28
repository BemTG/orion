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
    data.append(FP16x16 { mag: 15563, sign: false });
    data.append(FP16x16 { mag: 13277, sign: false });
    data.append(FP16x16 { mag: 70539, sign: false });
    data.append(FP16x16 { mag: 168337, sign: false });
    data.append(FP16x16 { mag: 10545, sign: true });
    data.append(FP16x16 { mag: 1401, sign: true });
    data.append(FP16x16 { mag: 108847, sign: false });
    data.append(FP16x16 { mag: 99628, sign: false });
    data.append(FP16x16 { mag: 16409, sign: false });
    data.append(FP16x16 { mag: 13978, sign: false });
    data.append(FP16x16 { mag: 105505, sign: false });
    data.append(FP16x16 { mag: 160889, sign: false });
    data.append(FP16x16 { mag: 11420, sign: true });
    data.append(FP16x16 { mag: 7467, sign: true });
    data.append(FP16x16 { mag: 71984, sign: false });
    data.append(FP16x16 { mag: 111468, sign: false });
    data.append(FP16x16 { mag: 10559, sign: false });
    data.append(FP16x16 { mag: 14008, sign: false });
    data.append(FP16x16 { mag: 129042, sign: false });
    data.append(FP16x16 { mag: 33813, sign: false });
    data.append(FP16x16 { mag: 5306, sign: true });
    data.append(FP16x16 { mag: 7118, sign: true });
    data.append(FP16x16 { mag: 85697, sign: false });
    data.append(FP16x16 { mag: 121054, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
