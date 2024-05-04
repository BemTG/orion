use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 128564, sign: false });
    data.append(FP16x16 { mag: 587269, sign: false });
    data.append(FP16x16 { mag: 200361, sign: false });
    data.append(FP16x16 { mag: 363639, sign: true });
    data.append(FP16x16 { mag: 239388, sign: false });
    data.append(FP16x16 { mag: 67492, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
