use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 448264, sign: false });
    data.append(FP16x16 { mag: 32282, sign: false });
    data.append(FP16x16 { mag: 466525, sign: false });
    data.append(FP16x16 { mag: 266179, sign: false });
    data.append(FP16x16 { mag: 17600, sign: false });
    data.append(FP16x16 { mag: 338088, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
