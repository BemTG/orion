use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(12);
    data.append(-13);
    data.append(13);
    data.append(-4);
    data.append(15);
    data.append(-7);
    data.append(12);
    data.append(15);
    data.append(6);
    TensorTrait::new(shape.span(), data.span())
}