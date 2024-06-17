use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(156);
    data.append(92);
    data.append(162);
    TensorTrait::new(shape.span(), data.span())
}
