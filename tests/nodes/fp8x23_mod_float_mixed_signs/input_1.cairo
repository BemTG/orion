use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 29498670, sign: true });
    data.append(FP8x23 { mag: 33110278, sign: true });
    data.append(FP8x23 { mag: 76067584, sign: true });
    data.append(FP8x23 { mag: 61065568, sign: true });
    data.append(FP8x23 { mag: 72202288, sign: false });
    data.append(FP8x23 { mag: 54036308, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
