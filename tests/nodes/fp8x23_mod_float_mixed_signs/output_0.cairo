use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 185532, sign: true });
    data.append(FP8x23 { mag: 19640278, sign: true });
    data.append(FP8x23 { mag: 12161784, sign: false });
    data.append(FP8x23 { mag: 1568096, sign: false });
    data.append(FP8x23 { mag: 13720208, sign: false });
    data.append(FP8x23 { mag: 7438304, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
