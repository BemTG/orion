use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(122);
    data.append(249);
    data.append(172);
    data.append(1);
    data.append(138);
    data.append(252);
    data.append(122);
    data.append(245);
    data.append(70);
    TensorTrait::new(shape.span(), data.span())
}
