use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 279404, sign: true });
    data.append(FP16x16 { mag: 94921, sign: true });
    data.append(FP16x16 { mag: 219389, sign: true });
    data.append(FP16x16 { mag: 536286, sign: true });
    data.append(FP16x16 { mag: 647771, sign: true });
    data.append(FP16x16 { mag: 452071, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
