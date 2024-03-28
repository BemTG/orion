use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8703131, sign: false });
    data.append(FP8x23 { mag: 12387906, sign: false });
    data.append(FP8x23 { mag: 630199, sign: false });
    data.append(FP8x23 { mag: 1358582, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
