use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(90);
    data.append(5);
    data.append(-101);
    data.append(-102);
    data.append(6);
    data.append(-12);
    data.append(87);
    data.append(105);
    data.append(57);
    TensorTrait::new(shape.span(), data.span())
}
