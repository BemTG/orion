use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 150849, sign: true });
    data.append(FP16x16 { mag: 95435, sign: false });
    data.append(FP16x16 { mag: 12039, sign: false });
    data.append(FP16x16 { mag: 15256, sign: true });
    data.append(FP16x16 { mag: 36325, sign: false });
    data.append(FP16x16 { mag: 54183, sign: true });
    data.append(FP16x16 { mag: 30995, sign: false });
    data.append(FP16x16 { mag: 8232, sign: true });
    data.append(FP16x16 { mag: 178595, sign: false });
    data.append(FP16x16 { mag: 17562, sign: true });
    data.append(FP16x16 { mag: 13348, sign: true });
    data.append(FP16x16 { mag: 24563, sign: true });
    data.append(FP16x16 { mag: 114342, sign: true });
    data.append(FP16x16 { mag: 15806, sign: false });
    data.append(FP16x16 { mag: 5561, sign: false });
    data.append(FP16x16 { mag: 23845, sign: true });
    data.append(FP16x16 { mag: 18205, sign: true });
    data.append(FP16x16 { mag: 125353, sign: false });
    data.append(FP16x16 { mag: 2292, sign: false });
    data.append(FP16x16 { mag: 31566, sign: true });
    data.append(FP16x16 { mag: 8924, sign: true });
    data.append(FP16x16 { mag: 113421, sign: true });
    data.append(FP16x16 { mag: 1176, sign: false });
    data.append(FP16x16 { mag: 31573, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
