use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(12);
    data.append(207);
    data.append(48);
    data.append(167);
    data.append(82);
    data.append(195);
    data.append(173);
    data.append(210);
    data.append(6);
    TensorTrait::new(shape.span(), data.span())
}
