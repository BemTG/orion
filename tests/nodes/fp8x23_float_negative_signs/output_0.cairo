use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1638504, sign: true });
    data.append(FP8x23 { mag: 7780836, sign: true });
    data.append(FP8x23 { mag: 19803248, sign: true });
    data.append(FP8x23 { mag: 27285106, sign: true });
    data.append(FP8x23 { mag: 16411236, sign: true });
    data.append(FP8x23 { mag: 3224453, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
