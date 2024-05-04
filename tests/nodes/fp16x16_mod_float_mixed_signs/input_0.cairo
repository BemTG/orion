use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 564189, sign: true });
    data.append(FP16x16 { mag: 33806, sign: true });
    data.append(FP16x16 { mag: 505516, sign: true });
    data.append(FP16x16 { mag: 19014, sign: false });
    data.append(FP16x16 { mag: 608611, sign: false });
    data.append(FP16x16 { mag: 21636, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
