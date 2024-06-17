use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(22178);
    data.append(63904);
    data.append(4775);
    data.append(29982);
    data.append(83432);
    data.append(8321);
    data.append(22004);
    data.append(60175);
    data.append(6773);
    TensorTrait::new(shape.span(), data.span())
}
