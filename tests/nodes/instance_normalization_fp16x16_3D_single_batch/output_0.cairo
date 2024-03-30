use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 51421, sign: false });
    data.append(FP16x16 { mag: 5988, sign: true });
    data.append(FP16x16 { mag: 60086, sign: false });
    data.append(FP16x16 { mag: 32582, sign: false });
    data.append(FP16x16 { mag: 42910, sign: false });
    data.append(FP16x16 { mag: 129209, sign: false });
    data.append(FP16x16 { mag: 6052, sign: false });
    data.append(FP16x16 { mag: 152130, sign: true });
    data.append(FP16x16 { mag: 32991, sign: true });
    data.append(FP16x16 { mag: 96809, sign: false });
    data.append(FP16x16 { mag: 15791, sign: false });
    data.append(FP16x16 { mag: 86302, sign: false });
    data.append(FP16x16 { mag: 122880, sign: true });
    data.append(FP16x16 { mag: 138814, sign: false });
    data.append(FP16x16 { mag: 86574, sign: true });
    data.append(FP16x16 { mag: 411631, sign: false });
    data.append(FP16x16 { mag: 53329, sign: true });
    data.append(FP16x16 { mag: 75141, sign: true });
    data.append(FP16x16 { mag: 59880, sign: true });
    data.append(FP16x16 { mag: 86524, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
