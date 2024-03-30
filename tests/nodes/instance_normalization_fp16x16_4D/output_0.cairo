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
    data.append(FP16x16 { mag: 148134, sign: false });
    data.append(FP16x16 { mag: 95756, sign: true });
    data.append(FP16x16 { mag: 60289, sign: false });
    data.append(FP16x16 { mag: 73034, sign: true });
    data.append(FP16x16 { mag: 145472, sign: true });
    data.append(FP16x16 { mag: 67859, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
