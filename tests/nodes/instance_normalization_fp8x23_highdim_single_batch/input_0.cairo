use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1496832, sign: false });
    data.append(FP8x23 { mag: 3957941, sign: true });
    data.append(FP8x23 { mag: 6658309, sign: true });
    data.append(FP8x23 { mag: 2774204, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
