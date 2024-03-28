use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 548162, sign: false });
    data.append(FP8x23 { mag: 547405, sign: false });
    data.append(FP8x23 { mag: 14568703, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
