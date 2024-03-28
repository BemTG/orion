use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16506599, sign: true });
    data.append(FP8x23 { mag: 10498581, sign: false });
    data.append(FP8x23 { mag: 19196028, sign: false });
    data.append(FP8x23 { mag: 4609985, sign: false });
    data.append(FP8x23 { mag: 15613983, sign: false });
    data.append(FP8x23 { mag: 2774380, sign: true });
    data.append(FP8x23 { mag: 6208105, sign: false });
    data.append(FP8x23 { mag: 1120176, sign: true });
    data.append(FP8x23 { mag: 5282879, sign: true });
    data.append(FP8x23 { mag: 35295096, sign: true });
    data.append(FP8x23 { mag: 2828770, sign: true });
    data.append(FP8x23 { mag: 39480976, sign: true });
    data.append(FP8x23 { mag: 11223668, sign: true });
    data.append(FP8x23 { mag: 20112944, sign: true });
    data.append(FP8x23 { mag: 14708194, sign: false });
    data.append(FP8x23 { mag: 9682216, sign: false });
    data.append(FP8x23 { mag: 24702640, sign: false });
    data.append(FP8x23 { mag: 2790661, sign: false });
    data.append(FP8x23 { mag: 15496213, sign: false });
    data.append(FP8x23 { mag: 12232412, sign: false });
    data.append(FP8x23 { mag: 24666200, sign: false });
    data.append(FP8x23 { mag: 11031577, sign: true });
    data.append(FP8x23 { mag: 14119696, sign: false });
    data.append(FP8x23 { mag: 3117310, sign: true });
    data.append(FP8x23 { mag: 8774969, sign: false });
    data.append(FP8x23 { mag: 19125236, sign: true });
    data.append(FP8x23 { mag: 7005324, sign: true });
    data.append(FP8x23 { mag: 13995188, sign: false });
    data.append(FP8x23 { mag: 26648658, sign: true });
    data.append(FP8x23 { mag: 519605, sign: false });
    data.append(FP8x23 { mag: 15223182, sign: true });
    data.append(FP8x23 { mag: 40179040, sign: true });
    data.append(FP8x23 { mag: 5855384, sign: false });
    data.append(FP8x23 { mag: 9250782, sign: false });
    data.append(FP8x23 { mag: 18642104, sign: true });
    data.append(FP8x23 { mag: 18724916, sign: false });
    data.append(FP8x23 { mag: 8521844, sign: false });
    data.append(FP8x23 { mag: 10666316, sign: false });
    data.append(FP8x23 { mag: 23501752, sign: false });
    data.append(FP8x23 { mag: 3489315, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
