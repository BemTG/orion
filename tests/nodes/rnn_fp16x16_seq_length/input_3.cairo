use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_3() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(10);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 148224, sign: true });
    data.append(FP16x16 { mag: 31632, sign: false });
    data.append(FP16x16 { mag: 93762, sign: true });
    data.append(FP16x16 { mag: 37117, sign: false });
    data.append(FP16x16 { mag: 31415, sign: false });
    data.append(FP16x16 { mag: 66786, sign: false });
    data.append(FP16x16 { mag: 22840, sign: false });
    data.append(FP16x16 { mag: 107467, sign: true });
    data.append(FP16x16 { mag: 128807, sign: false });
    data.append(FP16x16 { mag: 50944, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
