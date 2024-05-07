use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1039985, sign: true });
    data.append(FP8x23 { mag: 11843008, sign: true });
    data.append(FP8x23 { mag: 22754482, sign: true });
    data.append(FP8x23 { mag: 69589112, sign: true });
    data.append(FP8x23 { mag: 10935848, sign: true });
    data.append(FP8x23 { mag: 5268897, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
