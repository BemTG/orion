use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 176605, sign: false });
    data.append(FP16x16 { mag: 80579, sign: false });
    data.append(FP16x16 { mag: 19361, sign: false });
    data.append(FP16x16 { mag: 574157, sign: false });
    data.append(FP16x16 { mag: 3137, sign: false });
    data.append(FP16x16 { mag: 9477, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
