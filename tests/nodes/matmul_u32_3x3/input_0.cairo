use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(104);
    data.append(99);
    data.append(239);
    data.append(42);
    data.append(92);
    data.append(15);
    data.append(197);
    data.append(200);
    data.append(219);
    TensorTrait::new(shape.span(), data.span())
}
