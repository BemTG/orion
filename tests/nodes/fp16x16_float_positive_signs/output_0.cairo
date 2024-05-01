use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 90041, sign: false });
    data.append(FP16x16 { mag: 25656, sign: false });
    data.append(FP16x16 { mag: 197520, sign: false });
    data.append(FP16x16 { mag: 53702, sign: false });
    data.append(FP16x16 { mag: 11490, sign: false });
    data.append(FP16x16 { mag: 56103, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
