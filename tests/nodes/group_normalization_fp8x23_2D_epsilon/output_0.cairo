use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2223219, sign: true });
    data.append(FP8x23 { mag: 11991048, sign: false });
    data.append(FP8x23 { mag: 5904659, sign: true });
    data.append(FP8x23 { mag: 2213032, sign: true });
    data.append(FP8x23 { mag: 14225090, sign: true });
    data.append(FP8x23 { mag: 6372144, sign: false });
    data.append(FP8x23 { mag: 8435274, sign: false });
    data.append(FP8x23 { mag: 1796790, sign: false });
    data.append(FP8x23 { mag: 21064238, sign: true });
    data.append(FP8x23 { mag: 3170266, sign: false });
    data.append(FP8x23 { mag: 1146004, sign: false });
    data.append(FP8x23 { mag: 241481, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
