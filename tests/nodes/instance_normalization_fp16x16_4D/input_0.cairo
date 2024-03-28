use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 97406, sign: false });
    data.append(FP16x16 { mag: 2605, sign: true });
    data.append(FP16x16 { mag: 22969, sign: false });
    data.append(FP16x16 { mag: 67698, sign: false });
    data.append(FP16x16 { mag: 5662, sign: true });
    data.append(FP16x16 { mag: 29035, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
