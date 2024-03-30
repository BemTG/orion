use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 106838, sign: true });
    data.append(FP16x16 { mag: 69724, sign: true });
    data.append(FP16x16 { mag: 119051, sign: true });
    data.append(FP16x16 { mag: 188766, sign: true });
    data.append(FP16x16 { mag: 2415, sign: false });
    data.append(FP16x16 { mag: 5352, sign: false });
    data.append(FP16x16 { mag: 114816, sign: true });
    data.append(FP16x16 { mag: 93350, sign: true });
    data.append(FP16x16 { mag: 76920, sign: true });
    data.append(FP16x16 { mag: 80675, sign: true });
    data.append(FP16x16 { mag: 174846, sign: true });
    data.append(FP16x16 { mag: 151937, sign: true });
    data.append(FP16x16 { mag: 79689, sign: true });
    data.append(FP16x16 { mag: 14899, sign: false });
    data.append(FP16x16 { mag: 123322, sign: true });
    data.append(FP16x16 { mag: 12285, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
