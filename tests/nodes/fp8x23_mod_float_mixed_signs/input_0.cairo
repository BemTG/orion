use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 19044286, sign: false });
    data.append(FP8x23 { mag: 6718009, sign: false });
    data.append(FP8x23 { mag: 8706461, sign: true });
    data.append(FP8x23 { mag: 39001216, sign: true });
    data.append(FP8x23 { mag: 37686960, sign: false });
    data.append(FP8x23 { mag: 70354848, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
