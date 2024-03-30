use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 130753, sign: true });
    data.append(FP16x16 { mag: 42688, sign: true });
    data.append(FP16x16 { mag: 42095, sign: false });
    data.append(FP16x16 { mag: 1942, sign: true });
    data.append(FP16x16 { mag: 136334, sign: true });
    data.append(FP16x16 { mag: 14268, sign: false });
    data.append(FP16x16 { mag: 17531, sign: true });
    data.append(FP16x16 { mag: 14924, sign: false });
    data.append(FP16x16 { mag: 23756, sign: false });
    data.append(FP16x16 { mag: 12512, sign: true });
    data.append(FP16x16 { mag: 3245, sign: true });
    data.append(FP16x16 { mag: 141287, sign: true });
    data.append(FP16x16 { mag: 931, sign: false });
    data.append(FP16x16 { mag: 35663, sign: false });
    data.append(FP16x16 { mag: 130853, sign: true });
    data.append(FP16x16 { mag: 30416, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
