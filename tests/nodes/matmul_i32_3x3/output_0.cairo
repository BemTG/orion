use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(1804);
    data.append(802);
    data.append(-14973);
    data.append(2148);
    data.append(-4758);
    data.append(3948);
    data.append(1767);
    data.append(14865);
    data.append(-10911);
    TensorTrait::new(shape.span(), data.span())
}
