use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(233);
    data.append(90);
    data.append(6);
    TensorTrait::new(shape.span(), data.span())
}
