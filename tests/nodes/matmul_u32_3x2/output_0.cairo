use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(18077);
    data.append(14377);
    data.append(22905);
    data.append(14386);
    data.append(12683);
    data.append(11960);
    data.append(2992);
    data.append(2156);
    data.append(4920);
    TensorTrait::new(shape.span(), data.span())
}
