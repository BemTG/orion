use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 48436, sign: true });
    data.append(FP16x16 { mag: 61259, sign: true });
    data.append(FP16x16 { mag: 223618, sign: false });
    data.append(FP16x16 { mag: 111088, sign: false });
    data.append(FP16x16 { mag: 62245, sign: true });
    data.append(FP16x16 { mag: 73667, sign: true });
    data.append(FP16x16 { mag: 209539, sign: false });
    data.append(FP16x16 { mag: 166097, sign: false });
    data.append(FP16x16 { mag: 79351, sign: true });
    data.append(FP16x16 { mag: 5112, sign: true });
    data.append(FP16x16 { mag: 178287, sign: false });
    data.append(FP16x16 { mag: 117027, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
