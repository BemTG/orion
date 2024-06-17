use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(96);
    data.append(94);
    data.append(35);
    data.append(97);
    data.append(-10);
    data.append(-69);
    data.append(25);
    data.append(40);
    data.append(106);
    TensorTrait::new(shape.span(), data.span())
}
