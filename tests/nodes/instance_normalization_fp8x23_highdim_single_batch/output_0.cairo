use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15159200, sign: false });
    data.append(FP8x23 { mag: 13189858, sign: false });
    data.append(FP8x23 { mag: 21802260, sign: false });
    data.append(FP8x23 { mag: 6577921, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
