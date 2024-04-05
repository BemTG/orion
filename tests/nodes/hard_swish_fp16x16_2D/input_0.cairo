use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 86053, sign: false });
    data.append(FP16x16 { mag: 171703, sign: true });
    data.append(FP16x16 { mag: 161705, sign: false });
    data.append(FP16x16 { mag: 127706, sign: true });
    data.append(FP16x16 { mag: 178264, sign: true });
    data.append(FP16x16 { mag: 58513, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
