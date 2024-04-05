use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13618582, sign: true });
    data.append(FP8x23 { mag: 10292922, sign: false });
    data.append(FP8x23 { mag: 24837362, sign: true });
    data.append(FP8x23 { mag: 20811664, sign: false });
    data.append(FP8x23 { mag: 9327981, sign: false });
    data.append(FP8x23 { mag: 7858417, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
