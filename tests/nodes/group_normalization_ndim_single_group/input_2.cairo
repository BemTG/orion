use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7560, sign: true });
    data.append(FP16x16 { mag: 63930, sign: true });
    data.append(FP16x16 { mag: 2667, sign: false });
    data.append(FP16x16 { mag: 28343, sign: false });
    data.append(FP16x16 { mag: 6210, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
