use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 20942, sign: true });
    data.append(FP16x16 { mag: 78639, sign: false });
    data.append(FP16x16 { mag: 22253, sign: true });
    data.append(FP16x16 { mag: 169309, sign: false });
    data.append(FP16x16 { mag: 18899, sign: true });
    data.append(FP16x16 { mag: 2437, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
