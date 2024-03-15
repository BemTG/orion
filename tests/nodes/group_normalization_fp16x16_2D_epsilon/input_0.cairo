use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65037, sign: true });
    data.append(FP16x16 { mag: 21924, sign: false });
    data.append(FP16x16 { mag: 100749, sign: false });
    data.append(FP16x16 { mag: 8863, sign: true });
    data.append(FP16x16 { mag: 70072, sign: false });
    data.append(FP16x16 { mag: 20336, sign: false });
    data.append(FP16x16 { mag: 68162, sign: false });
    data.append(FP16x16 { mag: 19151, sign: true });
    data.append(FP16x16 { mag: 32368, sign: true });
    data.append(FP16x16 { mag: 54379, sign: false });
    data.append(FP16x16 { mag: 123485, sign: true });
    data.append(FP16x16 { mag: 84304, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
