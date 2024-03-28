use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 76024, sign: true });
    data.append(FP16x16 { mag: 26024, sign: false });
    data.append(FP16x16 { mag: 96125, sign: true });
    data.append(FP16x16 { mag: 14632, sign: true });
    data.append(FP16x16 { mag: 125434, sign: false });
    data.append(FP16x16 { mag: 26833, sign: true });
    data.append(FP16x16 { mag: 21530, sign: true });
    data.append(FP16x16 { mag: 120794, sign: true });
    data.append(FP16x16 { mag: 125432, sign: false });
    data.append(FP16x16 { mag: 26833, sign: true });
    data.append(FP16x16 { mag: 21529, sign: true });
    data.append(FP16x16 { mag: 120795, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
