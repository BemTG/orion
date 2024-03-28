use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 17361, sign: true });
    data.append(FP16x16 { mag: 29700, sign: false });
    data.append(FP16x16 { mag: 17284, sign: true });
    data.append(FP16x16 { mag: 8656, sign: true });
    data.append(FP16x16 { mag: 17068, sign: true });
    data.append(FP16x16 { mag: 29407, sign: false });
    data.append(FP16x16 { mag: 9902, sign: true });
    data.append(FP16x16 { mag: 16038, sign: true });
    data.append(FP16x16 { mag: 14453, sign: true });
    data.append(FP16x16 { mag: 26792, sign: false });
    data.append(FP16x16 { mag: 16106, sign: true });
    data.append(FP16x16 { mag: 9834, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
