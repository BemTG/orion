use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 176605, sign: false });
    data.append(FP16x16 { mag: 249358, sign: false });
    data.append(FP16x16 { mag: 264484, sign: false });
    data.append(FP16x16 { mag: 574157, sign: false });
    data.append(FP16x16 { mag: 527543, sign: false });
    data.append(FP16x16 { mag: 628628, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
