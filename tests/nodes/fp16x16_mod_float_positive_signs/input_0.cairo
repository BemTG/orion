use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 52001, sign: false });
    data.append(FP16x16 { mag: 634761, sign: false });
    data.append(FP16x16 { mag: 63216, sign: false });
    data.append(FP16x16 { mag: 375513, sign: false });
    data.append(FP16x16 { mag: 475621, sign: false });
    data.append(FP16x16 { mag: 266146, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
