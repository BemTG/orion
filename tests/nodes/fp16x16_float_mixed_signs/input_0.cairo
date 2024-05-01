use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 139673, sign: true });
    data.append(FP16x16 { mag: 617788, sign: false });
    data.append(FP16x16 { mag: 122739, sign: true });
    data.append(FP16x16 { mag: 184896, sign: false });
    data.append(FP16x16 { mag: 20298, sign: true });
    data.append(FP16x16 { mag: 535106, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
