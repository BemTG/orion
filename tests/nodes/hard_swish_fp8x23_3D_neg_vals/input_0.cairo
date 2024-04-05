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
    data.append(FP8x23 { mag: 47546164, sign: true });
    data.append(FP8x23 { mag: 29390966, sign: true });
    data.append(FP8x23 { mag: 49325968, sign: true });
    data.append(FP8x23 { mag: 47496036, sign: true });
    data.append(FP8x23 { mag: 45323512, sign: true });
    data.append(FP8x23 { mag: 44783832, sign: true });
    data.append(FP8x23 { mag: 40755224, sign: true });
    data.append(FP8x23 { mag: 30589474, sign: true });
    data.append(FP8x23 { mag: 49651456, sign: true });
    data.append(FP8x23 { mag: 29809138, sign: true });
    data.append(FP8x23 { mag: 37879540, sign: true });
    data.append(FP8x23 { mag: 30786086, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
