use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 311463, sign: true });
    data.append(FP16x16 { mag: 438613, sign: true });
    data.append(FP16x16 { mag: 450882, sign: true });
    data.append(FP16x16 { mag: 514384, sign: true });
    data.append(FP16x16 { mag: 606528, sign: true });
    data.append(FP16x16 { mag: 482247, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
