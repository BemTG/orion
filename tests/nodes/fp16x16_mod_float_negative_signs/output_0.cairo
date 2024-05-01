use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 20544, sign: true });
    data.append(FP16x16 { mag: 3148, sign: true });
    data.append(FP16x16 { mag: 4290, sign: true });
    data.append(FP16x16 { mag: 121008, sign: true });
    data.append(FP16x16 { mag: 71270, sign: true });
    data.append(FP16x16 { mag: 19029, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
