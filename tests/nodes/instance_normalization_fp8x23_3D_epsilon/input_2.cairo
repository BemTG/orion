use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5584808, sign: true });
    data.append(FP8x23 { mag: 12686512, sign: true });
    data.append(FP8x23 { mag: 1905977, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
