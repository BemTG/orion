use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4268948, sign: true });
    data.append(FP8x23 { mag: 15951068, sign: true });
    data.append(FP8x23 { mag: 57696672, sign: true });
    data.append(FP8x23 { mag: 36234028, sign: true });
    data.append(FP8x23 { mag: 8785868, sign: true });
    data.append(FP8x23 { mag: 52953200, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
