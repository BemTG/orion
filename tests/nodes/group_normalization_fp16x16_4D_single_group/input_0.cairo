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
    data.append(FP16x16 { mag: 45734, sign: true });
    data.append(FP16x16 { mag: 64488, sign: false });
    data.append(FP16x16 { mag: 25646, sign: false });
    data.append(FP16x16 { mag: 31205, sign: true });
    data.append(FP16x16 { mag: 22216, sign: false });
    data.append(FP16x16 { mag: 62737, sign: true });
    data.append(FP16x16 { mag: 37398, sign: true });
    data.append(FP16x16 { mag: 20628, sign: true });
    data.append(FP16x16 { mag: 38244, sign: false });
    data.append(FP16x16 { mag: 20604, sign: true });
    data.append(FP16x16 { mag: 3320, sign: true });
    data.append(FP16x16 { mag: 24432, sign: true });
    data.append(FP16x16 { mag: 55027, sign: false });
    data.append(FP16x16 { mag: 82786, sign: true });
    data.append(FP16x16 { mag: 51712, sign: false });
    data.append(FP16x16 { mag: 29427, sign: true });
    data.append(FP16x16 { mag: 60018, sign: true });
    data.append(FP16x16 { mag: 90623, sign: false });
    data.append(FP16x16 { mag: 52032, sign: false });
    data.append(FP16x16 { mag: 138219, sign: false });
    data.append(FP16x16 { mag: 58879, sign: false });
    data.append(FP16x16 { mag: 10428, sign: true });
    data.append(FP16x16 { mag: 61442, sign: true });
    data.append(FP16x16 { mag: 11590, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
