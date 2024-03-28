use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 70013, sign: true });
    data.append(FP16x16 { mag: 54499, sign: false });
    data.append(FP16x16 { mag: 19055, sign: false });
    data.append(FP16x16 { mag: 15441, sign: true });
    data.append(FP16x16 { mag: 5965, sign: false });
    data.append(FP16x16 { mag: 117161, sign: false });
    data.append(FP16x16 { mag: 72275, sign: false });
    data.append(FP16x16 { mag: 93451, sign: false });
    data.append(FP16x16 { mag: 65161, sign: true });
    data.append(FP16x16 { mag: 3105, sign: true });
    data.append(FP16x16 { mag: 38417, sign: true });
    data.append(FP16x16 { mag: 60192, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
