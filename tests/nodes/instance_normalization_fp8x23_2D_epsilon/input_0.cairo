use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 694065, sign: true });
    data.append(FP8x23 { mag: 9005771, sign: true });
    data.append(FP8x23 { mag: 10431446, sign: false });
    data.append(FP8x23 { mag: 2989875, sign: false });
    data.append(FP8x23 { mag: 8096813, sign: false });
    data.append(FP8x23 { mag: 951846, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
