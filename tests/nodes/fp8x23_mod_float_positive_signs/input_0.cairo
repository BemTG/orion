use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 77321224, sign: false });
    data.append(FP8x23 { mag: 32195702, sign: false });
    data.append(FP8x23 { mag: 73723768, sign: false });
    data.append(FP8x23 { mag: 2402193, sign: false });
    data.append(FP8x23 { mag: 53974864, sign: false });
    data.append(FP8x23 { mag: 63148224, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
