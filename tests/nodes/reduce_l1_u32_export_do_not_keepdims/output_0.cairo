use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(3);
    data.append(7);
    data.append(11);
    data.append(15);
    data.append(19);
    data.append(23);
    TensorTrait::new(shape.span(), data.span())
}
