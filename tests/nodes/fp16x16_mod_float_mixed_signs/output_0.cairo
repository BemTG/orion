use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 91723, sign: false });
    data.append(FP16x16 { mag: 59095, sign: true });
    data.append(FP16x16 { mag: 118618, sign: true });
    data.append(FP16x16 { mag: 233571, sign: true });
    data.append(FP16x16 { mag: 106545, sign: false });
    data.append(FP16x16 { mag: 39915, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
