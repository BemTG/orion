use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 26510610, sign: true });
    data.append(FP8x23 { mag: 9635217, sign: true });
    data.append(FP8x23 { mag: 38337568, sign: true });
    data.append(FP8x23 { mag: 58709596, sign: true });
    data.append(FP8x23 { mag: 36959572, sign: true });
    data.append(FP8x23 { mag: 19513440, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
