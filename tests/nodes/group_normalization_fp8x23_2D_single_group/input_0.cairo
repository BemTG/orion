use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14676620, sign: false });
    data.append(FP8x23 { mag: 4449554, sign: true });
    data.append(FP8x23 { mag: 2759204, sign: false });
    data.append(FP8x23 { mag: 1193736, sign: true });
    data.append(FP8x23 { mag: 863131, sign: false });
    data.append(FP8x23 { mag: 10681643, sign: false });
    data.append(FP8x23 { mag: 3907726, sign: true });
    data.append(FP8x23 { mag: 2156347, sign: true });
    data.append(FP8x23 { mag: 13564128, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
