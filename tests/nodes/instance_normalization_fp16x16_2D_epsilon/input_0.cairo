use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 110267, sign: true });
    data.append(FP16x16 { mag: 21954, sign: true });
    data.append(FP16x16 { mag: 51555, sign: true });
    data.append(FP16x16 { mag: 5720, sign: true });
    data.append(FP16x16 { mag: 124032, sign: true });
    data.append(FP16x16 { mag: 59307, sign: true });
    data.append(FP16x16 { mag: 29502, sign: true });
    data.append(FP16x16 { mag: 156059, sign: false });
    data.append(FP16x16 { mag: 85912, sign: false });
    data.append(FP16x16 { mag: 6460, sign: false });
    data.append(FP16x16 { mag: 7822, sign: true });
    data.append(FP16x16 { mag: 63917, sign: true });
    data.append(FP16x16 { mag: 114572, sign: false });
    data.append(FP16x16 { mag: 50232, sign: true });
    data.append(FP16x16 { mag: 166786, sign: false });
    data.append(FP16x16 { mag: 16339, sign: true });
    data.append(FP16x16 { mag: 126543, sign: true });
    data.append(FP16x16 { mag: 48145, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
