use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-15);
    data.append(47);
    data.append(-56);
    data.append(45);
    data.append(82);
    data.append(-108);
    data.append(-29);
    data.append(38);
    data.append(93);
    TensorTrait::new(shape.span(), data.span())
}
