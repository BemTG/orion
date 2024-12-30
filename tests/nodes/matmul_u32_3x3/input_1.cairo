use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn input_1() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(216);
    data.append(21);
    data.append(95);
    data.append(54);
    data.append(74);
    data.append(177);
    data.append(180);
    data.append(163);
    data.append(139);
    TensorTrait::new(shape.span(), data.span())
}
