use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 20687, sign: false });
    data.append(FP16x16 { mag: 81958, sign: true });
    data.append(FP16x16 { mag: 4836, sign: true });
    data.append(FP16x16 { mag: 36852, sign: false });
    data.append(FP16x16 { mag: 42597, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
