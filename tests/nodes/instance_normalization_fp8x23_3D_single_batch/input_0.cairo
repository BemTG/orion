use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1708673, sign: true });
    data.append(FP8x23 { mag: 3101113, sign: true });
    data.append(FP8x23 { mag: 15622224, sign: false });
    data.append(FP8x23 { mag: 8713601, sign: true });
    data.append(FP8x23 { mag: 2760521, sign: false });
    data.append(FP8x23 { mag: 17126388, sign: false });
    data.append(FP8x23 { mag: 26043974, sign: false });
    data.append(FP8x23 { mag: 288227, sign: false });
    data.append(FP8x23 { mag: 701499, sign: true });
    data.append(FP8x23 { mag: 6168910, sign: false });
    data.append(FP8x23 { mag: 4400739, sign: true });
    data.append(FP8x23 { mag: 3337725, sign: true });
    data.append(FP8x23 { mag: 5899549, sign: true });
    data.append(FP8x23 { mag: 2349459, sign: true });
    data.append(FP8x23 { mag: 4804306, sign: false });
    data.append(FP8x23 { mag: 20991708, sign: true });
    data.append(FP8x23 { mag: 7059514, sign: false });
    data.append(FP8x23 { mag: 2417841, sign: true });
    data.append(FP8x23 { mag: 2819097, sign: false });
    data.append(FP8x23 { mag: 1461887, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
