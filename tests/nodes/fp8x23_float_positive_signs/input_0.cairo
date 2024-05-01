use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 43748476, sign: false });
    data.append(FP8x23 { mag: 14592713, sign: false });
    data.append(FP8x23 { mag: 45157540, sign: false });
    data.append(FP8x23 { mag: 63934856, sign: false });
    data.append(FP8x23 { mag: 74241976, sign: false });
    data.append(FP8x23 { mag: 25799368, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
