use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 48642, sign: true });
    data.append(FP16x16 { mag: 17069, sign: true });
    data.append(FP16x16 { mag: 115237, sign: false });
    data.append(FP16x16 { mag: 75458, sign: false });
    data.append(FP16x16 { mag: 30184, sign: true });
    data.append(FP16x16 { mag: 48642, sign: true });
    data.append(FP16x16 { mag: 17069, sign: true });
    data.append(FP16x16 { mag: 115237, sign: false });
    data.append(FP16x16 { mag: 75458, sign: false });
    data.append(FP16x16 { mag: 30184, sign: true });
    data.append(FP16x16 { mag: 48642, sign: true });
    data.append(FP16x16 { mag: 17069, sign: true });
    data.append(FP16x16 { mag: 115237, sign: false });
    data.append(FP16x16 { mag: 75458, sign: false });
    data.append(FP16x16 { mag: 30184, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
