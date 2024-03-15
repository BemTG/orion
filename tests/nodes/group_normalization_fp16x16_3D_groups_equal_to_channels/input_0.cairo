use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 33957, sign: true });
    data.append(FP16x16 { mag: 10167, sign: true });
    data.append(FP16x16 { mag: 129509, sign: true });
    data.append(FP16x16 { mag: 67959, sign: true });
    data.append(FP16x16 { mag: 40788, sign: true });
    data.append(FP16x16 { mag: 58371, sign: false });
    data.append(FP16x16 { mag: 5496, sign: false });
    data.append(FP16x16 { mag: 65093, sign: false });
    data.append(FP16x16 { mag: 21216, sign: false });
    data.append(FP16x16 { mag: 17822, sign: true });
    data.append(FP16x16 { mag: 21414, sign: true });
    data.append(FP16x16 { mag: 26503, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
