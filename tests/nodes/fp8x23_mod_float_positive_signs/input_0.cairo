use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 76787512, sign: false });
    data.append(FP8x23 { mag: 10729195, sign: false });
    data.append(FP8x23 { mag: 33992544, sign: false });
    data.append(FP8x23 { mag: 83345960, sign: false });
    data.append(FP8x23 { mag: 17316852, sign: false });
    data.append(FP8x23 { mag: 58253972, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
