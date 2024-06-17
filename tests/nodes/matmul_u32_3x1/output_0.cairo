use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(22271);
    data.append(7424);
    data.append(15847);
    data.append(50509);
    data.append(16944);
    data.append(34643);
    data.append(37910);
    data.append(12528);
    data.append(28300);
    TensorTrait::new(shape.span(), data.span())
}
