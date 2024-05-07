use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 379833, sign: false });
    data.append(FP16x16 { mag: 84389, sign: false });
    data.append(FP16x16 { mag: 245122, sign: false });
    data.append(FP16x16 { mag: 679898, sign: false });
    data.append(FP16x16 { mag: 65550, sign: false });
    data.append(FP16x16 { mag: 619150, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
