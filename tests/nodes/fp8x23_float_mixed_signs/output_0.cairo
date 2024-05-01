use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3351246, sign: false });
    data.append(FP8x23 { mag: 1278762, sign: true });
    data.append(FP8x23 { mag: 11496368, sign: false });
    data.append(FP8x23 { mag: 7430384, sign: true });
    data.append(FP8x23 { mag: 38957200, sign: true });
    data.append(FP8x23 { mag: 63500332, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
