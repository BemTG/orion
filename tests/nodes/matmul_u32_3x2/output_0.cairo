use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(39960);
    data.append(42962);
    data.append(24420);
    data.append(45620);
    data.append(36879);
    data.append(14490);
    data.append(63060);
    data.append(56331);
    data.append(25920);
    TensorTrait::new(shape.span(), data.span())
}
