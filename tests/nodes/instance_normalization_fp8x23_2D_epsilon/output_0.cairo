use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 473187, sign: true });
    data.append(FP8x23 { mag: 11790551, sign: false });
    data.append(FP8x23 { mag: 378584, sign: false });
    data.append(FP8x23 { mag: 1793923, sign: true });
    data.append(FP8x23 { mag: 974844, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
