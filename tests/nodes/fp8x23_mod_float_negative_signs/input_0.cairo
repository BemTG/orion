use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 74262184, sign: true });
    data.append(FP8x23 { mag: 73413320, sign: true });
    data.append(FP8x23 { mag: 22754482, sign: true });
    data.append(FP8x23 { mag: 69589112, sign: true });
    data.append(FP8x23 { mag: 23316624, sign: true });
    data.append(FP8x23 { mag: 5268897, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
