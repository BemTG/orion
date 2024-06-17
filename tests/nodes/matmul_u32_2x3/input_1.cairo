use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(199);
    data.append(82);
    data.append(225);
    data.append(20);
    data.append(28);
    data.append(221);
    TensorTrait::new(shape.span(), data.span())
}
