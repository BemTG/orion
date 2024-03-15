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
    data.append(FP16x16 { mag: 47350, sign: true });
    data.append(FP16x16 { mag: 92090, sign: true });
    data.append(FP16x16 { mag: 8687, sign: true });
    data.append(FP16x16 { mag: 32487, sign: true });
    data.append(FP16x16 { mag: 192601, sign: false });
    data.append(FP16x16 { mag: 24400, sign: false });
    data.append(FP16x16 { mag: 115871, sign: false });
    data.append(FP16x16 { mag: 19083, sign: false });
    data.append(FP16x16 { mag: 106960, sign: true });
    data.append(FP16x16 { mag: 63081, sign: false });
    data.append(FP16x16 { mag: 111587, sign: true });
    data.append(FP16x16 { mag: 87525, sign: false });
    data.append(FP16x16 { mag: 68060, sign: false });
    data.append(FP16x16 { mag: 18631, sign: false });
    data.append(FP16x16 { mag: 118089, sign: false });
    data.append(FP16x16 { mag: 14263, sign: false });
    data.append(FP16x16 { mag: 90327, sign: false });
    data.append(FP16x16 { mag: 31712, sign: false });
    data.append(FP16x16 { mag: 69400, sign: false });
    data.append(FP16x16 { mag: 53489, sign: true });
    data.append(FP16x16 { mag: 38476, sign: true });
    data.append(FP16x16 { mag: 28301, sign: true });
    data.append(FP16x16 { mag: 5215, sign: false });
    data.append(FP16x16 { mag: 8342, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
