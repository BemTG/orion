use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 317556, sign: true });
    data.append(FP16x16 { mag: 538989, sign: false });
    data.append(FP16x16 { mag: 244494, sign: false });
    data.append(FP16x16 { mag: 385098, sign: true });
    data.append(FP16x16 { mag: 309371, sign: false });
    data.append(FP16x16 { mag: 321886, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
