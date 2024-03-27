use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 13640, sign: false });
    data.append(FP16x16 { mag: 4369, sign: true });
    data.append(FP16x16 { mag: 7925, sign: true });
    data.append(FP16x16 { mag: 149884, sign: true });
    data.append(FP16x16 { mag: 5151, sign: false });
    data.append(FP16x16 { mag: 9785, sign: false });
    data.append(FP16x16 { mag: 71282, sign: false });
    data.append(FP16x16 { mag: 76471, sign: false });
    data.append(FP16x16 { mag: 51209, sign: false });
    data.append(FP16x16 { mag: 30679, sign: true });
    data.append(FP16x16 { mag: 6498, sign: false });
    data.append(FP16x16 { mag: 30487, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
