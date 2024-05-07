use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 69380536, sign: true });
    data.append(FP8x23 { mag: 31435446, sign: true });
    data.append(FP8x23 { mag: 44864016, sign: true });
    data.append(FP8x23 { mag: 57873536, sign: false });
    data.append(FP8x23 { mag: 9837555, sign: false });
    data.append(FP8x23 { mag: 70682504, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
