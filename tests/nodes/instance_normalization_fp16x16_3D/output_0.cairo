use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 83403, sign: false });
    data.append(FP16x16 { mag: 150027, sign: false });
    data.append(FP16x16 { mag: 65695, sign: false });
    data.append(FP16x16 { mag: 79754, sign: true });
    data.append(FP16x16 { mag: 102872, sign: false });
    data.append(FP16x16 { mag: 94384, sign: true });
    data.append(FP16x16 { mag: 28222, sign: true });
    data.append(FP16x16 { mag: 32468, sign: true });
    data.append(FP16x16 { mag: 44750, sign: true });
    data.append(FP16x16 { mag: 80686, sign: true });
    data.append(FP16x16 { mag: 177147, sign: false });
    data.append(FP16x16 { mag: 80567, sign: false });
    data.append(FP16x16 { mag: 82468, sign: false });
    data.append(FP16x16 { mag: 63220, sign: true });
    data.append(FP16x16 { mag: 45281, sign: false });
    data.append(FP16x16 { mag: 37252, sign: true });
    data.append(FP16x16 { mag: 44701, sign: true });
    data.append(FP16x16 { mag: 49946, sign: true });
    data.append(FP16x16 { mag: 108573, sign: true });
    data.append(FP16x16 { mag: 40038, sign: true });
    data.append(FP16x16 { mag: 121048, sign: false });
    data.append(FP16x16 { mag: 58358, sign: true });
    data.append(FP16x16 { mag: 166968, sign: false });
    data.append(FP16x16 { mag: 62120, sign: false });
    data.append(FP16x16 { mag: 30466, sign: false });
    data.append(FP16x16 { mag: 47742, sign: true });
    data.append(FP16x16 { mag: 81670, sign: true });
    data.append(FP16x16 { mag: 92793, sign: true });
    data.append(FP16x16 { mag: 31611, sign: true });
    data.append(FP16x16 { mag: 26693, sign: true });
    data.append(FP16x16 { mag: 169143, sign: false });
    data.append(FP16x16 { mag: 71352, sign: false });
    data.append(FP16x16 { mag: 61436, sign: true });
    data.append(FP16x16 { mag: 32887, sign: false });
    data.append(FP16x16 { mag: 110298, sign: false });
    data.append(FP16x16 { mag: 57926, sign: true });
    data.append(FP16x16 { mag: 67041, sign: true });
    data.append(FP16x16 { mag: 16862, sign: true });
    data.append(FP16x16 { mag: 41761, sign: true });
    data.append(FP16x16 { mag: 96920, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
