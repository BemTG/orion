use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(60372);
    data.append(52859);
    data.append(73371);
    data.append(80514);
    data.append(41743);
    data.append(78478);
    data.append(50346);
    data.append(17257);
    data.append(47848);
    TensorTrait::new(shape.span(), data.span())
}
