use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3101, sign: true });
    data.append(FP16x16 { mag: 4287, sign: false });
    data.append(FP16x16 { mag: 16923, sign: false });
    data.append(FP16x16 { mag: 4113, sign: true });
    data.append(FP16x16 { mag: 13443, sign: false });
    data.append(FP16x16 { mag: 1846, sign: false });
    data.append(FP16x16 { mag: 12894, sign: false });
    data.append(FP16x16 { mag: 13645, sign: true });
    data.append(FP16x16 { mag: 49798, sign: false });
    data.append(FP16x16 { mag: 41233, sign: false });
    data.append(FP16x16 { mag: 32752, sign: false });
    data.append(FP16x16 { mag: 86976, sign: false });
    data.append(FP16x16 { mag: 56957, sign: false });
    data.append(FP16x16 { mag: 57864, sign: false });
    data.append(FP16x16 { mag: 39889, sign: false });
    data.append(FP16x16 { mag: 30031, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
