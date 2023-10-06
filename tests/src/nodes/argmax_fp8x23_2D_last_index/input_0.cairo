use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 847249408, sign: true });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    TensorTrait::new(shape.span(), data.span())
}