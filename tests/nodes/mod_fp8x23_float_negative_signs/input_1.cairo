use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 54585420, sign: true });
    data.append(FP8x23 { mag: 82558496, sign: true });
    data.append(FP8x23 { mag: 19482990, sign: true });
    data.append(FP8x23 { mag: 24012346, sign: true });
    data.append(FP8x23 { mag: 17680304, sign: true });
    data.append(FP8x23 { mag: 65062088, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
