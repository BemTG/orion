use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-2);
    data.append(-8);
    data.append(-6);
    data.append(4);
    data.append(16);
    data.append(12);
    data.append(1);
    data.append(4);
    data.append(3);
    TensorTrait::new(shape.span(), data.span())
}