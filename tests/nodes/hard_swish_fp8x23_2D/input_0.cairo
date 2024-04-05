use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7288980, sign: true });
    data.append(FP8x23 { mag: 10737083, sign: true });
    data.append(FP8x23 { mag: 13174853, sign: false });
    data.append(FP8x23 { mag: 3521607, sign: true });
    data.append(FP8x23 { mag: 16922174, sign: true });
    data.append(FP8x23 { mag: 20163618, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
