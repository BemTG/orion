use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-2058);
    data.append(789);
    data.append(669);
    data.append(-10446);
    data.append(-8277);
    data.append(-8121);
    data.append(-12852);
    data.append(-6974);
    data.append(-6982);
    TensorTrait::new(shape.span(), data.span())
}
