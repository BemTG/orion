use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 301982, sign: false });
    data.append(FP16x16 { mag: 261221, sign: false });
    data.append(FP16x16 { mag: 241155, sign: false });
    data.append(FP16x16 { mag: 177479, sign: false });
    data.append(FP16x16 { mag: 445255, sign: false });
    data.append(FP16x16 { mag: 561726, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
