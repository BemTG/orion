use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2690672, sign: true });
    data.append(FP8x23 { mag: 57193452, sign: true });
    data.append(FP8x23 { mag: 3774374, sign: true });
    data.append(FP8x23 { mag: 10117240, sign: true });
    data.append(FP8x23 { mag: 11810941, sign: true });
    data.append(FP8x23 { mag: 25115014, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
