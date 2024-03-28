use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 16156, sign: false });
    data.append(FP16x16 { mag: 197, sign: true });
    data.append(FP16x16 { mag: 25032, sign: false });
    data.append(FP16x16 { mag: 35657, sign: true });
    data.append(FP16x16 { mag: 78291, sign: false });
    data.append(FP16x16 { mag: 52547, sign: false });
    data.append(FP16x16 { mag: 13151, sign: true });
    data.append(FP16x16 { mag: 27505, sign: true });
    data.append(FP16x16 { mag: 24676, sign: false });
    data.append(FP16x16 { mag: 45788, sign: true });
    data.append(FP16x16 { mag: 6312, sign: true });
    data.append(FP16x16 { mag: 39198, sign: true });
    data.append(FP16x16 { mag: 43748, sign: false });
    data.append(FP16x16 { mag: 47117, sign: true });
    data.append(FP16x16 { mag: 24774, sign: true });
    data.append(FP16x16 { mag: 69429, sign: false });
    data.append(FP16x16 { mag: 13500, sign: false });
    data.append(FP16x16 { mag: 65368, sign: true });
    data.append(FP16x16 { mag: 14790, sign: true });
    data.append(FP16x16 { mag: 47489, sign: false });
    data.append(FP16x16 { mag: 31165, sign: false });
    data.append(FP16x16 { mag: 18512, sign: true });
    data.append(FP16x16 { mag: 9728, sign: false });
    data.append(FP16x16 { mag: 6247, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
