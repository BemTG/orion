use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 174739, sign: true });
    data.append(FP16x16 { mag: 227654, sign: true });
    data.append(FP16x16 { mag: 69668, sign: true });
    data.append(FP16x16 { mag: 64911, sign: true });
    data.append(FP16x16 { mag: 210651, sign: true });
    data.append(FP16x16 { mag: 182594, sign: true });
    data.append(FP16x16 { mag: 62543, sign: true });
    data.append(FP16x16 { mag: 71180, sign: true });
    data.append(FP16x16 { mag: 138889, sign: true });
    data.append(FP16x16 { mag: 101685, sign: true });
    data.append(FP16x16 { mag: 66973, sign: true });
    data.append(FP16x16 { mag: 52470, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
