use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(93);
    data.append(46);
    data.append(150);
    data.append(125);
    data.append(106);
    data.append(43);
    data.append(136);
    data.append(23);
    data.append(190);
    TensorTrait::new(shape.span(), data.span())
}
