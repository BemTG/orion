use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 418405, sign: false });
    data.append(FP16x16 { mag: 606680, sign: false });
    data.append(FP16x16 { mag: 611070, sign: false });
    data.append(FP16x16 { mag: 534949, sign: false });
    data.append(FP16x16 { mag: 603657, sign: false });
    data.append(FP16x16 { mag: 633862, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
