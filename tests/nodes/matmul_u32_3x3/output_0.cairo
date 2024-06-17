use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(72803);
    data.append(81792);
    data.append(55443);
    data.append(66654);
    data.append(64443);
    data.append(28470);
    data.append(54489);
    data.append(60044);
    data.append(54051);
    TensorTrait::new(shape.span(), data.span())
}
