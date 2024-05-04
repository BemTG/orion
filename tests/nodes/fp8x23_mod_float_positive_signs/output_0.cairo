use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 20293344, sign: false });
    data.append(FP8x23 { mag: 34930696, sign: false });
    data.append(FP8x23 { mag: 67261616, sign: false });
    data.append(FP8x23 { mag: 6527724, sign: false });
    data.append(FP8x23 { mag: 1112076, sign: false });
    data.append(FP8x23 { mag: 37391152, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
