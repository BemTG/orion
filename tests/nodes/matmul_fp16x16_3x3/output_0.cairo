use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    TensorTrait::new(shape.span(), data.span())
}