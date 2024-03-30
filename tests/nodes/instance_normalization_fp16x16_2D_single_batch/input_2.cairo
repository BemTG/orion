use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 81594, sign: false });
    data.append(FP16x16 { mag: 86008, sign: false });
    data.append(FP16x16 { mag: 38039, sign: true });
    data.append(FP16x16 { mag: 16521, sign: false });
    data.append(FP16x16 { mag: 83708, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
