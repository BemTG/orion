use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 26155982, sign: true });
    data.append(FP8x23 { mag: 36769628, sign: true });
    data.append(FP8x23 { mag: 45543188, sign: true });
    data.append(FP8x23 { mag: 14227834, sign: true });
    data.append(FP8x23 { mag: 12566968, sign: true });
    data.append(FP8x23 { mag: 2470308, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
