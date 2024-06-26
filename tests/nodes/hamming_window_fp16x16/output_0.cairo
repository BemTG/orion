use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5696, sign: false });
    data.append(FP16x16 { mag: 12688, sign: false });
    data.append(FP16x16 { mag: 30400, sign: false });
    data.append(FP16x16 { mag: 50560, sign: false });
    data.append(FP16x16 { mag: 63712, sign: false });
    data.append(FP16x16 { mag: 63744, sign: false });
    data.append(FP16x16 { mag: 50624, sign: false });
    data.append(FP16x16 { mag: 30432, sign: false });
    data.append(FP16x16 { mag: 12752, sign: false });
    data.append(FP16x16 { mag: 5696, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
