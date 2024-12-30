use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-1258);
    data.append(1156);
    data.append(-4046);
    data.append(3737);
    data.append(-3434);
    data.append(12019);
    data.append(-2553);
    data.append(2346);
    data.append(-8211);
    TensorTrait::new(shape.span(), data.span())
}
