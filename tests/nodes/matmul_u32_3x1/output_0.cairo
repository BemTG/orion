use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(7922);
    data.append(3060);
    data.append(204);
    data.append(42872);
    data.append(16560);
    data.append(1104);
    data.append(14446);
    data.append(5580);
    data.append(372);
    TensorTrait::new(shape.span(), data.span())
}
