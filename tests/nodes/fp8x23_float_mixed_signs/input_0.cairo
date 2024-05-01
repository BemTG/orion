use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3351246, sign: false });
    data.append(FP8x23 { mag: 52873572, sign: true });
    data.append(FP8x23 { mag: 66547616, sign: false });
    data.append(FP8x23 { mag: 28005142, sign: true });
    data.append(FP8x23 { mag: 38957200, sign: true });
    data.append(FP8x23 { mag: 63500332, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
