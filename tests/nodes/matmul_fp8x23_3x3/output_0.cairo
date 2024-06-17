use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: true });
    data.append(FP8x23 { mag: 41943040, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
