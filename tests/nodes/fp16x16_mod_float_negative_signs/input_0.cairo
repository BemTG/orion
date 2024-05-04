use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 587343, sign: true });
    data.append(FP16x16 { mag: 579400, sign: true });
    data.append(FP16x16 { mag: 475889, sign: true });
    data.append(FP16x16 { mag: 197853, sign: true });
    data.append(FP16x16 { mag: 612650, sign: true });
    data.append(FP16x16 { mag: 318474, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
