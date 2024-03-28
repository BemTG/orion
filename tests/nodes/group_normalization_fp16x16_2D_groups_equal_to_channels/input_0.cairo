use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 46402, sign: true });
    data.append(FP16x16 { mag: 91424, sign: true });
    data.append(FP16x16 { mag: 36692, sign: false });
    data.append(FP16x16 { mag: 50787, sign: true });
    data.append(FP16x16 { mag: 780, sign: false });
    data.append(FP16x16 { mag: 76741, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
