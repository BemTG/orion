use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 43438696, sign: true });
    data.append(FP8x23 { mag: 39214680, sign: true });
    data.append(FP8x23 { mag: 50823264, sign: true });
    data.append(FP8x23 { mag: 74126104, sign: true });
    data.append(FP8x23 { mag: 66643816, sign: true });
    data.append(FP8x23 { mag: 31162816, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
