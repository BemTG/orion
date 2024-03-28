use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12593354, sign: false });
    data.append(FP8x23 { mag: 14340979, sign: true });
    data.append(FP8x23 { mag: 17993718, sign: false });
    data.append(FP8x23 { mag: 7197425, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
