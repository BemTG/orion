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
    data.append(FP16x16 { mag: 26093, sign: true });
    data.append(FP16x16 { mag: 135791, sign: true });
    data.append(FP16x16 { mag: 57277, sign: true });
    data.append(FP16x16 { mag: 33, sign: false });
    data.append(FP16x16 { mag: 20716, sign: false });
    data.append(FP16x16 { mag: 182602, sign: true });
    data.append(FP16x16 { mag: 56984, sign: true });
    data.append(FP16x16 { mag: 259, sign: true });
    data.append(FP16x16 { mag: 156487, sign: true });
    data.append(FP16x16 { mag: 5398, sign: true });
    data.append(FP16x16 { mag: 54751, sign: true });
    data.append(FP16x16 { mag: 2493, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
