use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2588907, sign: true });
    data.append(FP8x23 { mag: 3078035, sign: true });
    data.append(FP8x23 { mag: 10036087, sign: false });
    data.append(FP8x23 { mag: 1514403, sign: true });
    data.append(FP8x23 { mag: 2771625, sign: true });
    data.append(FP8x23 { mag: 18159658, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
