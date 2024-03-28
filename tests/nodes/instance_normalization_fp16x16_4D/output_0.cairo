use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 169434, sign: true });
    data.append(FP16x16 { mag: 143542, sign: false });
    data.append(FP16x16 { mag: 63507, sign: false });
    data.append(FP16x16 { mag: 18978, sign: false });
    data.append(FP16x16 { mag: 24044, sign: true });
    data.append(FP16x16 { mag: 3695, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
