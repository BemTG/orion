use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 82591, sign: true });
    data.append(FP16x16 { mag: 174570, sign: true });
    data.append(FP16x16 { mag: 35677, sign: false });
    data.append(FP16x16 { mag: 23719, sign: true });
    data.append(FP16x16 { mag: 30176, sign: false });
    data.append(FP16x16 { mag: 18854, sign: false });
    data.append(FP16x16 { mag: 82591, sign: true });
    data.append(FP16x16 { mag: 174570, sign: true });
    data.append(FP16x16 { mag: 35677, sign: false });
    data.append(FP16x16 { mag: 23719, sign: true });
    data.append(FP16x16 { mag: 30176, sign: false });
    data.append(FP16x16 { mag: 18854, sign: false });
    data.append(FP16x16 { mag: 82591, sign: true });
    data.append(FP16x16 { mag: 174570, sign: true });
    data.append(FP16x16 { mag: 35677, sign: false });
    data.append(FP16x16 { mag: 23719, sign: true });
    data.append(FP16x16 { mag: 30176, sign: false });
    data.append(FP16x16 { mag: 18854, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
