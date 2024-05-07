use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 182730, sign: false });
    data.append(FP16x16 { mag: 74626, sign: true });
    data.append(FP16x16 { mag: 423899, sign: false });
    data.append(FP16x16 { mag: 385356, sign: true });
    data.append(FP16x16 { mag: 640775, sign: false });
    data.append(FP16x16 { mag: 195802, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
