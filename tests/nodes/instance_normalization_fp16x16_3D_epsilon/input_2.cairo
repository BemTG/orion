use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54514, sign: false });
    data.append(FP16x16 { mag: 4261, sign: true });
    data.append(FP16x16 { mag: 75416, sign: false });
    data.append(FP16x16 { mag: 117560, sign: false });
    data.append(FP16x16 { mag: 44742, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
