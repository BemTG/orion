use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 44496, sign: true });
    data.append(FP16x16 { mag: 49042, sign: true });
    data.append(FP16x16 { mag: 38489, sign: false });
    data.append(FP16x16 { mag: 5093, sign: true });
    data.append(FP16x16 { mag: 13282, sign: true });
    data.append(FP16x16 { mag: 11440, sign: false });
    data.append(FP16x16 { mag: 22575, sign: false });
    data.append(FP16x16 { mag: 75402, sign: false });
    data.append(FP16x16 { mag: 56782, sign: false });
    data.append(FP16x16 { mag: 59071, sign: false });
    data.append(FP16x16 { mag: 34975, sign: false });
    data.append(FP16x16 { mag: 38683, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
