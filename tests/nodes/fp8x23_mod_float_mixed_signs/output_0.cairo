use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12764510, sign: false });
    data.append(FP8x23 { mag: 16358532, sign: true });
    data.append(FP8x23 { mag: 34720516, sign: false });
    data.append(FP8x23 { mag: 51249536, sign: true });
    data.append(FP8x23 { mag: 10812422, sign: true });
    data.append(FP8x23 { mag: 47582304, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
