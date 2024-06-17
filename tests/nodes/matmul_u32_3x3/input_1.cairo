use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(98);
    data.append(144);
    data.append(125);
    data.append(221);
    data.append(0);
    data.append(169);
    data.append(66);
    data.append(77);
    data.append(159);
    TensorTrait::new(shape.span(), data.span())
}
