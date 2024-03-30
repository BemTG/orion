use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3750953, sign: false });
    data.append(FP8x23 { mag: 2243134, sign: true });
    data.append(FP8x23 { mag: 2942079, sign: false });
    data.append(FP8x23 { mag: 2128466, sign: false });
    data.append(FP8x23 { mag: 815924, sign: false });
    data.append(FP8x23 { mag: 5028149, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
