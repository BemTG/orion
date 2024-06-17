use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(123);
    data.append(19);
    data.append(248);
    data.append(63);
    data.append(241);
    data.append(12);
    TensorTrait::new(shape.span(), data.span())
}
