use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13283098, sign: true });
    data.append(FP8x23 { mag: 10997588, sign: true });
    data.append(FP8x23 { mag: 1519536, sign: false });
    data.append(FP8x23 { mag: 7240271, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
