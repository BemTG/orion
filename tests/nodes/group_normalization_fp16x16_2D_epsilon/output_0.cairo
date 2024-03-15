use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38133, sign: true });
    data.append(FP16x16 { mag: 24552, sign: false });
    data.append(FP16x16 { mag: 87890, sign: true });
    data.append(FP16x16 { mag: 110647, sign: true });
    data.append(FP16x16 { mag: 138864, sign: false });
    data.append(FP16x16 { mag: 3839, sign: true });
    data.append(FP16x16 { mag: 87053, sign: true });
    data.append(FP16x16 { mag: 108889, sign: true });
    data.append(FP16x16 { mag: 38090, sign: true });
    data.append(FP16x16 { mag: 24545, sign: false });
    data.append(FP16x16 { mag: 45439, sign: true });
    data.append(FP16x16 { mag: 21452, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
