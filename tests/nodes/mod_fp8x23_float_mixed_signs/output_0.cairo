use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14640538, sign: true });
    data.append(FP8x23 { mag: 512216, sign: true });
    data.append(FP8x23 { mag: 44864016, sign: true });
    data.append(FP8x23 { mag: 12919820, sign: false });
    data.append(FP8x23 { mag: 9837555, sign: false });
    data.append(FP8x23 { mag: 14810792, sign: true });
    TensorTrait::new(shape.span(), data.span())
}