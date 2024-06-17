use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-1408);
    data.append(-2560);
    data.append(-6592);
    data.append(594);
    data.append(1080);
    data.append(2781);
    data.append(2442);
    data.append(4440);
    data.append(11433);
    TensorTrait::new(shape.span(), data.span())
}
