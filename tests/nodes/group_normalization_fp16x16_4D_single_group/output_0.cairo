use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112560, sign: false });
    data.append(FP16x16 { mag: 39239, sign: false });
    data.append(FP16x16 { mag: 65078, sign: false });
    data.append(FP16x16 { mag: 102896, sign: false });
    data.append(FP16x16 { mag: 10690, sign: true });
    data.append(FP16x16 { mag: 86930, sign: true });
    data.append(FP16x16 { mag: 64190, sign: true });
    data.append(FP16x16 { mag: 49140, sign: true });
    data.append(FP16x16 { mag: 64840, sign: false });
    data.append(FP16x16 { mag: 100558, sign: false });
    data.append(FP16x16 { mag: 90067, sign: false });
    data.append(FP16x16 { mag: 102881, sign: false });
    data.append(FP16x16 { mag: 6449, sign: false });
    data.append(FP16x16 { mag: 106394, sign: true });
    data.append(FP16x16 { mag: 3735, sign: false });
    data.append(FP16x16 { mag: 62703, sign: true });
    data.append(FP16x16 { mag: 123186, sign: false });
    data.append(FP16x16 { mag: 62698, sign: false });
    data.append(FP16x16 { mag: 78193, sign: false });
    data.append(FP16x16 { mag: 43586, sign: false });
    data.append(FP16x16 { mag: 21597, sign: true });
    data.append(FP16x16 { mag: 59142, sign: true });
    data.append(FP16x16 { mag: 86777, sign: true });
    data.append(FP16x16 { mag: 59772, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
