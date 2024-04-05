use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 44925, sign: true });
    data.append(FP16x16 { mag: 54749, sign: false });
    data.append(FP16x16 { mag: 193308, sign: false });
    data.append(FP16x16 { mag: 115828, sign: false });
    data.append(FP16x16 { mag: 134417, sign: true });
    data.append(FP16x16 { mag: 1800, sign: true });
    data.append(FP16x16 { mag: 144749, sign: false });
    data.append(FP16x16 { mag: 118384, sign: true });
    data.append(FP16x16 { mag: 191635, sign: false });
    data.append(FP16x16 { mag: 28020, sign: false });
    data.append(FP16x16 { mag: 101147, sign: false });
    data.append(FP16x16 { mag: 86887, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
