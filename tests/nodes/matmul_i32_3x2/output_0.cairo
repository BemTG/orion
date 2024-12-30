use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-14502);
    data.append(8992);
    data.append(-76);
    data.append(17266);
    data.append(-8421);
    data.append(-677);
    data.append(16304);
    data.append(-9721);
    data.append(-45);
    TensorTrait::new(shape.span(), data.span())
}
