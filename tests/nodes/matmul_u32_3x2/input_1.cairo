use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(82);
    data.append(250);
    data.append(9);
    data.append(92);
    data.append(241);
    data.append(35);
    TensorTrait::new(shape.span(), data.span())
}