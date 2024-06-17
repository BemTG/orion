use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-110);
    data.append(93);
    data.append(-76);
    data.append(-37);
    data.append(-117);
    data.append(-43);
    data.append(-81);
    data.append(-43);
    data.append(-24);
    TensorTrait::new(shape.span(), data.span())
}
