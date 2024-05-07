use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 63251408, sign: false });
    data.append(FP8x23 { mag: 7677666, sign: false });
    data.append(FP8x23 { mag: 1585248, sign: false });
    data.append(FP8x23 { mag: 11308176, sign: true });
    data.append(FP8x23 { mag: 2728816, sign: false });
    data.append(FP8x23 { mag: 29499920, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
