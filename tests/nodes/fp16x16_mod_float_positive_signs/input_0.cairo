use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 148135, sign: false });
    data.append(FP16x16 { mag: 96363, sign: false });
    data.append(FP16x16 { mag: 432974, sign: false });
    data.append(FP16x16 { mag: 87584, sign: false });
    data.append(FP16x16 { mag: 299834, sign: false });
    data.append(FP16x16 { mag: 153787, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
