use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 45540864, sign: false });
    data.append(FP8x23 { mag: 53451852, sign: false });
    data.append(FP8x23 { mag: 56590356, sign: false });
    data.append(FP8x23 { mag: 44925640, sign: false });
    data.append(FP8x23 { mag: 11347671, sign: false });
    data.append(FP8x23 { mag: 36686080, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
