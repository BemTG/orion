use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 65831776, sign: true });
    data.append(FP8x23 { mag: 20832396, sign: false });
    data.append(FP8x23 { mag: 39248340, sign: false });
    data.append(FP8x23 { mag: 13099584, sign: false });
    data.append(FP8x23 { mag: 11576168, sign: true });
    data.append(FP8x23 { mag: 24939230, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
