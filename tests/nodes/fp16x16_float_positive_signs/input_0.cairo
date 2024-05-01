use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 538306, sign: false });
    data.append(FP16x16 { mag: 380760, sign: false });
    data.append(FP16x16 { mag: 197520, sign: false });
    data.append(FP16x16 { mag: 586060, sign: false });
    data.append(FP16x16 { mag: 222694, sign: false });
    data.append(FP16x16 { mag: 394192, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
