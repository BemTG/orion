use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 70471, sign: true });
    data.append(FP16x16 { mag: 18374, sign: false });
    data.append(FP16x16 { mag: 15286, sign: false });
    data.append(FP16x16 { mag: 28273, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
