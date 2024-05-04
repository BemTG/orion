use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 314127, sign: false });
    data.append(FP8x23 { mag: 20091558, sign: false });
    data.append(FP8x23 { mag: 17274330, sign: false });
    data.append(FP8x23 { mag: 23693854, sign: false });
    data.append(FP8x23 { mag: 60630960, sign: true });
    data.append(FP8x23 { mag: 39058284, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
