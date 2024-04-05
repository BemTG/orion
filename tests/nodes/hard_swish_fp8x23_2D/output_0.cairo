use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3124417, sign: true });
    data.append(FP8x23 { mag: 7251384, sign: false });
    data.append(FP8x23 { mag: 162086, sign: true });
    data.append(FP8x23 { mag: 19011260, sign: false });
    data.append(FP8x23 { mag: 6392748, sign: false });
    data.append(FP8x23 { mag: 2702252, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
