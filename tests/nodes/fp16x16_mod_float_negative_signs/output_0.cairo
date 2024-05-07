use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112950, sign: true });
    data.append(FP16x16 { mag: 70221, sign: true });
    data.append(FP16x16 { mag: 96553, sign: true });
    data.append(FP16x16 { mag: 229113, sign: true });
    data.append(FP16x16 { mag: 214147, sign: true });
    data.append(FP16x16 { mag: 145311, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
