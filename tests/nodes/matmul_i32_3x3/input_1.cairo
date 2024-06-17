use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(45);
    data.append(112);
    data.append(26);
    data.append(51);
    data.append(-8);
    data.append(-71);
    data.append(11);
    data.append(85);
    data.append(38);
    TensorTrait::new(shape.span(), data.span())
}
