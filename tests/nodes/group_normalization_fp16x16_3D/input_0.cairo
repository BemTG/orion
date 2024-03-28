use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 46155, sign: true });
    data.append(FP16x16 { mag: 67955, sign: true });
    data.append(FP16x16 { mag: 54063, sign: true });
    data.append(FP16x16 { mag: 106476, sign: true });
    data.append(FP16x16 { mag: 40183, sign: true });
    data.append(FP16x16 { mag: 21234, sign: false });
    data.append(FP16x16 { mag: 52360, sign: true });
    data.append(FP16x16 { mag: 35186, sign: true });
    data.append(FP16x16 { mag: 22446, sign: false });
    data.append(FP16x16 { mag: 24326, sign: true });
    data.append(FP16x16 { mag: 47602, sign: true });
    data.append(FP16x16 { mag: 107507, sign: true });
    data.append(FP16x16 { mag: 83720, sign: true });
    data.append(FP16x16 { mag: 24766, sign: true });
    data.append(FP16x16 { mag: 54783, sign: false });
    data.append(FP16x16 { mag: 108549, sign: true });
    data.append(FP16x16 { mag: 43650, sign: true });
    data.append(FP16x16 { mag: 38875, sign: false });
    data.append(FP16x16 { mag: 22441, sign: true });
    data.append(FP16x16 { mag: 105646, sign: false });
    data.append(FP16x16 { mag: 39561, sign: false });
    data.append(FP16x16 { mag: 20473, sign: false });
    data.append(FP16x16 { mag: 32917, sign: false });
    data.append(FP16x16 { mag: 70361, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
