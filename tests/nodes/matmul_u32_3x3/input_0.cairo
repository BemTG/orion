use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(42);
    data.append(220);
    data.append(219);
    data.append(212);
    data.append(233);
    data.append(123);
    data.append(186);
    data.append(165);
    data.append(7);
    TensorTrait::new(shape.span(), data.span())
}
