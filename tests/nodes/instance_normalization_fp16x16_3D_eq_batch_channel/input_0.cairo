use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 59963, sign: true });
    data.append(FP16x16 { mag: 120607, sign: true });
    data.append(FP16x16 { mag: 40007, sign: true });
    data.append(FP16x16 { mag: 73902, sign: false });
    data.append(FP16x16 { mag: 8495, sign: true });
    data.append(FP16x16 { mag: 10925, sign: true });
    data.append(FP16x16 { mag: 88501, sign: false });
    data.append(FP16x16 { mag: 70740, sign: false });
    data.append(FP16x16 { mag: 63526, sign: true });
    data.append(FP16x16 { mag: 60183, sign: true });
    data.append(FP16x16 { mag: 23679, sign: false });
    data.append(FP16x16 { mag: 3278, sign: false });
    data.append(FP16x16 { mag: 38643, sign: false });
    data.append(FP16x16 { mag: 80831, sign: true });
    data.append(FP16x16 { mag: 93754, sign: false });
    data.append(FP16x16 { mag: 46493, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
