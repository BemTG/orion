use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-7485);
    data.append(4656);
    data.append(-4692);
    data.append(6378);
    data.append(-801);
    data.append(-327);
    data.append(13960);
    data.append(7013);
    data.append(8387);
    TensorTrait::new(shape.span(), data.span())
}
