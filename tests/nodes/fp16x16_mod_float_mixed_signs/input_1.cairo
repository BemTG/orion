use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 645032, sign: true });
    data.append(FP16x16 { mag: 645670, sign: false });
    data.append(FP16x16 { mag: 237134, sign: true });
    data.append(FP16x16 { mag: 487813, sign: true });
    data.append(FP16x16 { mag: 519909, sign: false });
    data.append(FP16x16 { mag: 435124, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
