use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 26924, sign: false });
    data.append(FP16x16 { mag: 275, sign: true });
    data.append(FP16x16 { mag: 21535, sign: true });
    data.append(FP16x16 { mag: 138780, sign: false });
    data.append(FP16x16 { mag: 28782, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
