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
    data.append(FP16x16 { mag: 214425, sign: true });
    data.append(FP16x16 { mag: 248551, sign: true });
    data.append(FP16x16 { mag: 390085, sign: true });
    data.append(FP16x16 { mag: 383358, sign: true });
    data.append(FP16x16 { mag: 293438, sign: true });
    data.append(FP16x16 { mag: 252985, sign: true });
    data.append(FP16x16 { mag: 202285, sign: true });
    data.append(FP16x16 { mag: 207385, sign: true });
    data.append(FP16x16 { mag: 392692, sign: true });
    data.append(FP16x16 { mag: 342928, sign: true });
    data.append(FP16x16 { mag: 339595, sign: true });
    data.append(FP16x16 { mag: 367387, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
