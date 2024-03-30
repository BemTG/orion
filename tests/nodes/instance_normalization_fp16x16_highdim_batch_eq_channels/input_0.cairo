use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 67811, sign: true });
    data.append(FP16x16 { mag: 121023, sign: true });
    data.append(FP16x16 { mag: 59424, sign: false });
    data.append(FP16x16 { mag: 45034, sign: true });
    data.append(FP16x16 { mag: 3049, sign: true });
    data.append(FP16x16 { mag: 87835, sign: false });
    data.append(FP16x16 { mag: 17944, sign: false });
    data.append(FP16x16 { mag: 146635, sign: false });
    data.append(FP16x16 { mag: 2766, sign: false });
    data.append(FP16x16 { mag: 10069, sign: false });
    data.append(FP16x16 { mag: 68248, sign: true });
    data.append(FP16x16 { mag: 105956, sign: false });
    data.append(FP16x16 { mag: 15529, sign: false });
    data.append(FP16x16 { mag: 48493, sign: true });
    data.append(FP16x16 { mag: 4843, sign: false });
    data.append(FP16x16 { mag: 25633, sign: true });
    data.append(FP16x16 { mag: 49398, sign: false });
    data.append(FP16x16 { mag: 15936, sign: true });
    data.append(FP16x16 { mag: 104201, sign: true });
    data.append(FP16x16 { mag: 25655, sign: false });
    data.append(FP16x16 { mag: 9755, sign: true });
    data.append(FP16x16 { mag: 25667, sign: false });
    data.append(FP16x16 { mag: 122959, sign: true });
    data.append(FP16x16 { mag: 40878, sign: true });
    data.append(FP16x16 { mag: 101507, sign: false });
    data.append(FP16x16 { mag: 62683, sign: false });
    data.append(FP16x16 { mag: 4351, sign: false });
    data.append(FP16x16 { mag: 43444, sign: true });
    data.append(FP16x16 { mag: 22139, sign: true });
    data.append(FP16x16 { mag: 41716, sign: true });
    data.append(FP16x16 { mag: 9213, sign: false });
    data.append(FP16x16 { mag: 352, sign: true });
    data.append(FP16x16 { mag: 99505, sign: false });
    data.append(FP16x16 { mag: 43358, sign: false });
    data.append(FP16x16 { mag: 76659, sign: true });
    data.append(FP16x16 { mag: 92006, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
