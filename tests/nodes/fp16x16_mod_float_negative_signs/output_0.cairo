use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 169027, sign: true });
    data.append(FP16x16 { mag: 579400, sign: true });
    data.append(FP16x16 { mag: 1851, sign: true });
    data.append(FP16x16 { mag: 197853, sign: true });
    data.append(FP16x16 { mag: 21452, sign: true });
    data.append(FP16x16 { mag: 7751, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
