use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(9499);
    data.append(12975);
    data.append(-2848);
    data.append(3096);
    data.append(5079);
    data.append(610);
    data.append(4331);
    data.append(11490);
    data.append(1838);
    TensorTrait::new(shape.span(), data.span())
}
