use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 34525, sign: false });
    data.append(FP16x16 { mag: 6510, sign: false });
    data.append(FP16x16 { mag: 41478, sign: false });
    data.append(FP16x16 { mag: 85247, sign: false });
    data.append(FP16x16 { mag: 68719, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
