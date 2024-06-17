use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(4264);
    data.append(3484);
    data.append(2704);
    data.append(-5330);
    data.append(-4355);
    data.append(-3380);
    data.append(-1230);
    data.append(-1005);
    data.append(-780);
    TensorTrait::new(shape.span(), data.span())
}
