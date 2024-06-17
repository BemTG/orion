use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(34);
    data.append(32);
    data.append(158);
    data.append(248);
    data.append(232);
    data.append(202);
    data.append(78);
    data.append(9);
    data.append(88);
    TensorTrait::new(shape.span(), data.span())
}
