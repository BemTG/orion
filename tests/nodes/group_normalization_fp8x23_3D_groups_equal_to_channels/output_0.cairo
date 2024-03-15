use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 30368276, sign: true });
    data.append(FP8x23 { mag: 13922443, sign: true });
    data.append(FP8x23 { mag: 10089162, sign: false });
    data.append(FP8x23 { mag: 24178906, sign: false });
    data.append(FP8x23 { mag: 15851940, sign: true });
    data.append(FP8x23 { mag: 28438780, sign: true });
    data.append(FP8x23 { mag: 11317650, sign: false });
    data.append(FP8x23 { mag: 22950418, sign: false });
    data.append(FP8x23 { mag: 28191662, sign: true });
    data.append(FP8x23 { mag: 16099056, sign: true });
    data.append(FP8x23 { mag: 24465768, sign: false });
    data.append(FP8x23 { mag: 9802299, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
