use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 146159, sign: false });
    data.append(FP8x23 { mag: 14230082, sign: false });
    data.append(FP8x23 { mag: 4800995, sign: true });
    data.append(FP8x23 { mag: 24788640, sign: true });
    data.append(FP8x23 { mag: 9976223, sign: false });
    data.append(FP8x23 { mag: 13599259, sign: false });
    data.append(FP8x23 { mag: 23878700, sign: true });
    data.append(FP8x23 { mag: 10623696, sign: false });
    data.append(FP8x23 { mag: 13747354, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
