use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5626782, sign: false });
    data.append(FP8x23 { mag: 10675713, sign: true });
    data.append(FP8x23 { mag: 3412050, sign: true });
    data.append(FP8x23 { mag: 6169912, sign: false });
    data.append(FP8x23 { mag: 3999904, sign: false });
    data.append(FP8x23 { mag: 4713845, sign: true });
    data.append(FP8x23 { mag: 15047346, sign: false });
    data.append(FP8x23 { mag: 5422235, sign: true });
    data.append(FP8x23 { mag: 2148993, sign: true });
    data.append(FP8x23 { mag: 5372099, sign: true });
    data.append(FP8x23 { mag: 3125494, sign: true });
    data.append(FP8x23 { mag: 8289249, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
