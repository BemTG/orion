use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(28650);
    data.append(8590);
    data.append(36496);
    data.append(79536);
    data.append(40646);
    data.append(85556);
    data.append(20347);
    data.append(6566);
    data.append(28807);
    TensorTrait::new(shape.span(), data.span())
}
