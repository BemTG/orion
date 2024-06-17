use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(32032);
    data.append(42416);
    data.append(21648);
    data.append(6188);
    data.append(8194);
    data.append(4182);
    data.append(20566);
    data.append(27233);
    data.append(13899);
    TensorTrait::new(shape.span(), data.span())
}
