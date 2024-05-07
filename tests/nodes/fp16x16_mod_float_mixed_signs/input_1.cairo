use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 309629, sign: true });
    data.append(FP16x16 { mag: 102374, sign: false });
    data.append(FP16x16 { mag: 55101, sign: false });
    data.append(FP16x16 { mag: 401783, sign: true });
    data.append(FP16x16 { mag: 545508, sign: true });
    data.append(FP16x16 { mag: 174704, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
