use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 46442456, sign: false });
    data.append(FP8x23 { mag: 6496013, sign: false });
    data.append(FP8x23 { mag: 54742288, sign: false });
    data.append(FP8x23 { mag: 18698338, sign: false });
    data.append(FP8x23 { mag: 11749912, sign: false });
    data.append(FP8x23 { mag: 45530660, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
