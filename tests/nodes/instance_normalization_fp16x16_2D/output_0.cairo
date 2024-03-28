use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 8460, sign: false });
    data.append(FP16x16 { mag: 87928, sign: true });
    data.append(FP16x16 { mag: 67875, sign: true });
    data.append(FP16x16 { mag: 8091, sign: true });
    data.append(FP16x16 { mag: 36147, sign: false });
    data.append(FP16x16 { mag: 8460, sign: false });
    data.append(FP16x16 { mag: 87928, sign: true });
    data.append(FP16x16 { mag: 67875, sign: true });
    data.append(FP16x16 { mag: 8091, sign: true });
    data.append(FP16x16 { mag: 36147, sign: false });
    data.append(FP16x16 { mag: 8460, sign: false });
    data.append(FP16x16 { mag: 87928, sign: true });
    data.append(FP16x16 { mag: 67875, sign: true });
    data.append(FP16x16 { mag: 8091, sign: true });
    data.append(FP16x16 { mag: 36147, sign: false });
    data.append(FP16x16 { mag: 8460, sign: false });
    data.append(FP16x16 { mag: 87928, sign: true });
    data.append(FP16x16 { mag: 67875, sign: true });
    data.append(FP16x16 { mag: 8091, sign: true });
    data.append(FP16x16 { mag: 36147, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
