use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 64903, sign: true });
    data.append(FP16x16 { mag: 24748, sign: true });
    data.append(FP16x16 { mag: 2161, sign: false });
    data.append(FP16x16 { mag: 114271, sign: true });
    data.append(FP16x16 { mag: 16162, sign: false });
    data.append(FP16x16 { mag: 33242, sign: true });
    data.append(FP16x16 { mag: 47963, sign: true });
    data.append(FP16x16 { mag: 39090, sign: false });
    data.append(FP16x16 { mag: 7762, sign: false });
    data.append(FP16x16 { mag: 61488, sign: false });
    data.append(FP16x16 { mag: 115702, sign: false });
    data.append(FP16x16 { mag: 19355, sign: true });
    data.append(FP16x16 { mag: 65815, sign: false });
    data.append(FP16x16 { mag: 41136, sign: false });
    data.append(FP16x16 { mag: 73970, sign: true });
    data.append(FP16x16 { mag: 75332, sign: false });
    data.append(FP16x16 { mag: 23040, sign: true });
    data.append(FP16x16 { mag: 594, sign: false });
    data.append(FP16x16 { mag: 40640, sign: true });
    data.append(FP16x16 { mag: 115190, sign: false });
    data.append(FP16x16 { mag: 96576, sign: true });
    data.append(FP16x16 { mag: 98479, sign: true });
    data.append(FP16x16 { mag: 62128, sign: true });
    data.append(FP16x16 { mag: 14817, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
