use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 691766, sign: false });
    data.append(FP16x16 { mag: 239602, sign: false });
    data.append(FP16x16 { mag: 512745, sign: false });
    data.append(FP16x16 { mag: 400031, sign: false });
    data.append(FP16x16 { mag: 486587, sign: false });
    data.append(FP16x16 { mag: 600291, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
