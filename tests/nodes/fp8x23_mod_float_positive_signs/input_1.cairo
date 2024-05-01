use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 81293384, sign: false });
    data.append(FP8x23 { mag: 81809656, sign: false });
    data.append(FP8x23 { mag: 32292160, sign: false });
    data.append(FP8x23 { mag: 74500744, sign: false });
    data.append(FP8x23 { mag: 31992388, sign: false });
    data.append(FP8x23 { mag: 40577448, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
