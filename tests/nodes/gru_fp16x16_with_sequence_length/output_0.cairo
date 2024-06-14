use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12890, sign: false });
    data.append(FP16x16 { mag: 46371, sign: true });
    data.append(FP16x16 { mag: 20392, sign: true });
    data.append(FP16x16 { mag: 7335, sign: true });
    data.append(FP16x16 { mag: 4126, sign: false });
    data.append(FP16x16 { mag: 19348, sign: false });
    data.append(FP16x16 { mag: 13252, sign: true });
    data.append(FP16x16 { mag: 338, sign: true });
    data.append(FP16x16 { mag: 534, sign: true });
    data.append(FP16x16 { mag: 2264, sign: false });
    data.append(FP16x16 { mag: 19761, sign: false });
    data.append(FP16x16 { mag: 34976, sign: false });
    data.append(FP16x16 { mag: 2, sign: true });
    data.append(FP16x16 { mag: 35, sign: true });
    data.append(FP16x16 { mag: 1225, sign: false });
    data.append(FP16x16 { mag: 24777, sign: false });
    data.append(FP16x16 { mag: 55656, sign: false });
    data.append(FP16x16 { mag: 20392, sign: true });
    data.append(FP16x16 { mag: 7337, sign: true });
    data.append(FP16x16 { mag: 4469, sign: false });
    data.append(FP16x16 { mag: 33804, sign: false });
    data.append(FP16x16 { mag: 64022, sign: false });
    data.append(FP16x16 { mag: 338, sign: true });
    data.append(FP16x16 { mag: 535, sign: true });
    data.append(FP16x16 { mag: 2629, sign: false });
    data.append(FP16x16 { mag: 30078, sign: false });
    data.append(FP16x16 { mag: 65233, sign: false });
    data.append(FP16x16 { mag: 2, sign: true });
    data.append(FP16x16 { mag: 35, sign: true });
    data.append(FP16x16 { mag: 1524, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24777, sign: false });
    data.append(FP16x16 { mag: 55656, sign: false });
    data.append(FP16x16 { mag: 20392, sign: true });
    data.append(FP16x16 { mag: 7337, sign: true });
    data.append(FP16x16 { mag: 4469, sign: false });
    data.append(FP16x16 { mag: 33804, sign: false });
    data.append(FP16x16 { mag: 64022, sign: false });
    data.append(FP16x16 { mag: 338, sign: true });
    data.append(FP16x16 { mag: 535, sign: true });
    data.append(FP16x16 { mag: 2629, sign: false });
    data.append(FP16x16 { mag: 30078, sign: false });
    data.append(FP16x16 { mag: 65233, sign: false });
    data.append(FP16x16 { mag: 2, sign: true });
    data.append(FP16x16 { mag: 35, sign: true });
    data.append(FP16x16 { mag: 1524, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
