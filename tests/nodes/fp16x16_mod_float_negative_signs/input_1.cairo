use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 119745, sign: true });
    data.append(FP16x16 { mag: 278519, sign: true });
    data.append(FP16x16 { mag: 62995, sign: true });
    data.append(FP16x16 { mag: 336142, sign: true });
    data.append(FP16x16 { mag: 654501, sign: true });
    data.append(FP16x16 { mag: 525800, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
