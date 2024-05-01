use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 88260, sign: true });
    data.append(FP16x16 { mag: 13145, sign: true });
    data.append(FP16x16 { mag: 130708, sign: true });
    data.append(FP16x16 { mag: 260002, sign: true });
    data.append(FP16x16 { mag: 458678, sign: true });
    data.append(FP16x16 { mag: 643386, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
