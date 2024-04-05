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
    data.append(FP16x16 { mag: 59068, sign: false });
    data.append(FP16x16 { mag: 100101, sign: false });
    data.append(FP16x16 { mag: 24381, sign: true });
    data.append(FP16x16 { mag: 11859, sign: true });
    data.append(FP16x16 { mag: 21883, sign: true });
    data.append(FP16x16 { mag: 32229, sign: false });
    data.append(FP16x16 { mag: 24321, sign: true });
    data.append(FP16x16 { mag: 14454, sign: true });
    data.append(FP16x16 { mag: 115453, sign: false });
    data.append(FP16x16 { mag: 56383, sign: false });
    data.append(FP16x16 { mag: 22717, sign: false });
    data.append(FP16x16 { mag: 14833, sign: false });
    data.append(FP16x16 { mag: 1173, sign: false });
    data.append(FP16x16 { mag: 20658, sign: true });
    data.append(FP16x16 { mag: 22431, sign: true });
    data.append(FP16x16 { mag: 18162, sign: true });
    data.append(FP16x16 { mag: 23682, sign: false });
    data.append(FP16x16 { mag: 72248, sign: false });
    data.append(FP16x16 { mag: 135252, sign: false });
    data.append(FP16x16 { mag: 22528, sign: true });
    data.append(FP16x16 { mag: 22513, sign: true });
    data.append(FP16x16 { mag: 91689, sign: false });
    data.append(FP16x16 { mag: 117352, sign: false });
    data.append(FP16x16 { mag: 115179, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
