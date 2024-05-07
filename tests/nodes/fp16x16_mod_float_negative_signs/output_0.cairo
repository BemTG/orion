use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112061, sign: true });
    data.append(FP16x16 { mag: 15154, sign: true });
    data.append(FP16x16 { mag: 15439, sign: true });
    data.append(FP16x16 { mag: 291295, sign: true });
    data.append(FP16x16 { mag: 13573, sign: true });
    data.append(FP16x16 { mag: 391103, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
