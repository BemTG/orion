use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 13834, sign: false });
    data.append(FP16x16 { mag: 94011, sign: false });
    data.append(FP16x16 { mag: 93123, sign: false });
    data.append(FP16x16 { mag: 967, sign: true });
    data.append(FP16x16 { mag: 100042, sign: true });
    data.append(FP16x16 { mag: 149622, sign: true });
    data.append(FP16x16 { mag: 60985, sign: false });
    data.append(FP16x16 { mag: 160840, sign: false });
    data.append(FP16x16 { mag: 40671, sign: true });
    data.append(FP16x16 { mag: 22601, sign: true });
    data.append(FP16x16 { mag: 64788, sign: false });
    data.append(FP16x16 { mag: 141695, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
