use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1970341, sign: false });
    data.append(FP8x23 { mag: 10201207, sign: true });
    data.append(FP8x23 { mag: 7636929, sign: true });
    data.append(FP8x23 { mag: 5483075, sign: true });
    data.append(FP8x23 { mag: 3459633, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
