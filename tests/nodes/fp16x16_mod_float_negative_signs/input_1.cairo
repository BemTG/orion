use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 418316, sign: true });
    data.append(FP16x16 { mag: 595693, sign: true });
    data.append(FP16x16 { mag: 118509, sign: true });
    data.append(FP16x16 { mag: 551596, sign: true });
    data.append(FP16x16 { mag: 295598, sign: true });
    data.append(FP16x16 { mag: 310723, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
