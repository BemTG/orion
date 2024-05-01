use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 138589, sign: false });
    data.append(FP16x16 { mag: 424370, sign: false });
    data.append(FP16x16 { mag: 201207, sign: false });
    data.append(FP16x16 { mag: 445373, sign: false });
    data.append(FP16x16 { mag: 631339, sign: false });
    data.append(FP16x16 { mag: 680193, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
