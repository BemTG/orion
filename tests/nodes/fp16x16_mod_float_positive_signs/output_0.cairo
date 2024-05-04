use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 160721, sign: false });
    data.append(FP16x16 { mag: 34957, sign: false });
    data.append(FP16x16 { mag: 336351, sign: false });
    data.append(FP16x16 { mag: 199858, sign: false });
    data.append(FP16x16 { mag: 163679, sign: false });
    data.append(FP16x16 { mag: 60689, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
