use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(-4);
    data.append(-3);
    data.append(4);
    data.append(1);
    data.append(3);
    data.append(-3);
    TensorTrait::new(shape.span(), data.span())
}