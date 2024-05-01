use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 78630008, sign: true });
    data.append(FP8x23 { mag: 24540490, sign: true });
    data.append(FP8x23 { mag: 40568700, sign: true });
    data.append(FP8x23 { mag: 13289176, sign: true });
    data.append(FP8x23 { mag: 28364472, sign: false });
    data.append(FP8x23 { mag: 43780840, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
