use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 83840, sign: false });
    data.append(FP16x16 { mag: 99428, sign: false });
    data.append(FP16x16 { mag: 30579, sign: true });
    data.append(FP16x16 { mag: 23420, sign: true });
    data.append(FP16x16 { mag: 50850, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
