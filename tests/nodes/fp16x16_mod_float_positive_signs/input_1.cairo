use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 463008, sign: false });
    data.append(FP16x16 { mag: 327001, sign: false });
    data.append(FP16x16 { mag: 384123, sign: false });
    data.append(FP16x16 { mag: 222500, sign: false });
    data.append(FP16x16 { mag: 568265, sign: false });
    data.append(FP16x16 { mag: 389554, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
