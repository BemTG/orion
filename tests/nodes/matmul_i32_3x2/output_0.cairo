use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(427);
    data.append(9225);
    data.append(-3564);
    data.append(5144);
    data.append(-3799);
    data.append(12976);
    data.append(-7927);
    data.append(-3306);
    data.append(-15540);
    TensorTrait::new(shape.span(), data.span())
}
