use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 526587, sign: true });
    data.append(FP16x16 { mag: 654539, sign: true });
    data.append(FP16x16 { mag: 228282, sign: true });
    data.append(FP16x16 { mag: 291295, sign: true });
    data.append(FP16x16 { mag: 226653, sign: true });
    data.append(FP16x16 { mag: 391103, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
