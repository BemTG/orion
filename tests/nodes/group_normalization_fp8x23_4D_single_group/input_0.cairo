use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3784459, sign: true });
    data.append(FP8x23 { mag: 3150893, sign: true });
    data.append(FP8x23 { mag: 3011642, sign: true });
    data.append(FP8x23 { mag: 7464795, sign: true });
    data.append(FP8x23 { mag: 3606498, sign: false });
    data.append(FP8x23 { mag: 3060649, sign: true });
    data.append(FP8x23 { mag: 8103035, sign: false });
    data.append(FP8x23 { mag: 2223044, sign: false });
    data.append(FP8x23 { mag: 9264300, sign: true });
    data.append(FP8x23 { mag: 22993268, sign: false });
    data.append(FP8x23 { mag: 422343, sign: false });
    data.append(FP8x23 { mag: 3693471, sign: true });
    data.append(FP8x23 { mag: 5990059, sign: false });
    data.append(FP8x23 { mag: 1488502, sign: true });
    data.append(FP8x23 { mag: 1703736, sign: false });
    data.append(FP8x23 { mag: 17934492, sign: false });
    data.append(FP8x23 { mag: 3901031, sign: true });
    data.append(FP8x23 { mag: 15745043, sign: true });
    data.append(FP8x23 { mag: 365014, sign: false });
    data.append(FP8x23 { mag: 16120403, sign: true });
    data.append(FP8x23 { mag: 9397760, sign: false });
    data.append(FP8x23 { mag: 6152338, sign: true });
    data.append(FP8x23 { mag: 5233578, sign: false });
    data.append(FP8x23 { mag: 5486189, sign: true });
    data.append(FP8x23 { mag: 14990538, sign: false });
    data.append(FP8x23 { mag: 3133780, sign: false });
    data.append(FP8x23 { mag: 21416256, sign: true });
    data.append(FP8x23 { mag: 790905, sign: true });
    data.append(FP8x23 { mag: 7846306, sign: false });
    data.append(FP8x23 { mag: 18099324, sign: false });
    data.append(FP8x23 { mag: 12619366, sign: false });
    data.append(FP8x23 { mag: 1261642, sign: false });
    data.append(FP8x23 { mag: 1821532, sign: true });
    data.append(FP8x23 { mag: 733683, sign: false });
    data.append(FP8x23 { mag: 3800314, sign: false });
    data.append(FP8x23 { mag: 5559168, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
