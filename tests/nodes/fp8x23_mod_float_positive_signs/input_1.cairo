use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 82826136, sign: false });
    data.append(FP8x23 { mag: 11652784, sign: false });
    data.append(FP8x23 { mag: 84430968, sign: false });
    data.append(FP8x23 { mag: 80394720, sign: false });
    data.append(FP8x23 { mag: 46916192, sign: false });
    data.append(FP8x23 { mag: 90243448, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
