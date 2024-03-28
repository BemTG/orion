use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1146078, sign: true });
    data.append(FP8x23 { mag: 10394674, sign: true });
    data.append(FP8x23 { mag: 5238098, sign: true });
    data.append(FP8x23 { mag: 10425978, sign: false });
    data.append(FP8x23 { mag: 9353295, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
