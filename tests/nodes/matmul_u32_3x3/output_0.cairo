use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(47845);
    data.append(33379);
    data.append(67732);
    data.append(25438);
    data.append(7203);
    data.append(23183);
    data.append(77960);
    data.append(45231);
    data.append(93246);
    TensorTrait::new(shape.span(), data.span())
}
