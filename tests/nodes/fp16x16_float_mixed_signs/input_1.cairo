use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 81811, sign: true });
    data.append(FP16x16 { mag: 200804, sign: true });
    data.append(FP16x16 { mag: 499799, sign: true });
    data.append(FP16x16 { mag: 568765, sign: false });
    data.append(FP16x16 { mag: 362880, sign: true });
    data.append(FP16x16 { mag: 242254, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
