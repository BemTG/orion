use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(31647);
    data.append(3003);
    data.append(47586);
    data.append(16851);
    data.append(1599);
    data.append(25338);
    data.append(26030);
    data.append(2470);
    data.append(39140);
    TensorTrait::new(shape.span(), data.span())
}
