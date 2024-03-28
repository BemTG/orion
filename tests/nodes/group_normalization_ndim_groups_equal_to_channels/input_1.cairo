use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 40777, sign: false });
    data.append(FP16x16 { mag: 30602, sign: false });
    data.append(FP16x16 { mag: 6424, sign: true });
    data.append(FP16x16 { mag: 41291, sign: true });
    data.append(FP16x16 { mag: 115277, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
