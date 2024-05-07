use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 21988144, sign: false });
    data.append(FP8x23 { mag: 45772208, sign: false });
    data.append(FP8x23 { mag: 38481092, sign: false });
    data.append(FP8x23 { mag: 38037736, sign: false });
    data.append(FP8x23 { mag: 31305400, sign: false });
    data.append(FP8x23 { mag: 42038952, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
