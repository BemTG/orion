use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13452926, sign: false });
    data.append(FP8x23 { mag: 17027660, sign: false });
    data.append(FP8x23 { mag: 51192188, sign: false });
    data.append(FP8x23 { mag: 40338140, sign: false });
    data.append(FP8x23 { mag: 6866278, sign: false });
    data.append(FP8x23 { mag: 19052272, sign: false });
    TensorTrait::new(shape.span(), data.span())
}