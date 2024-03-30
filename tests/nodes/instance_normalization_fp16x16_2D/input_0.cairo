use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 36327, sign: true });
    data.append(FP16x16 { mag: 36746, sign: true });
    data.append(FP16x16 { mag: 145249, sign: true });
    data.append(FP16x16 { mag: 39904, sign: false });
    data.append(FP16x16 { mag: 111677, sign: false });
    data.append(FP16x16 { mag: 72957, sign: true });
    data.append(FP16x16 { mag: 10993, sign: false });
    data.append(FP16x16 { mag: 31963, sign: false });
    data.append(FP16x16 { mag: 101894, sign: true });
    data.append(FP16x16 { mag: 56473, sign: false });
    data.append(FP16x16 { mag: 37406, sign: false });
    data.append(FP16x16 { mag: 66871, sign: false });
    data.append(FP16x16 { mag: 20402, sign: true });
    data.append(FP16x16 { mag: 115854, sign: true });
    data.append(FP16x16 { mag: 44639, sign: true });
    data.append(FP16x16 { mag: 10628, sign: true });
    data.append(FP16x16 { mag: 10851, sign: true });
    data.append(FP16x16 { mag: 10659, sign: false });
    data.append(FP16x16 { mag: 20635, sign: false });
    data.append(FP16x16 { mag: 65979, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
