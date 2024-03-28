use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 45205, sign: true });
    data.append(FP16x16 { mag: 46420, sign: false });
    data.append(FP16x16 { mag: 42305, sign: true });
    data.append(FP16x16 { mag: 48291, sign: false });
    data.append(FP16x16 { mag: 105742, sign: false });
    data.append(FP16x16 { mag: 34691, sign: false });
    data.append(FP16x16 { mag: 1897, sign: false });
    data.append(FP16x16 { mag: 51895, sign: true });
    data.append(FP16x16 { mag: 25247, sign: false });
    data.append(FP16x16 { mag: 22987, sign: true });
    data.append(FP16x16 { mag: 53264, sign: false });
    data.append(FP16x16 { mag: 50012, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
