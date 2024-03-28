use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16181675, sign: false });
    data.append(FP8x23 { mag: 4050731, sign: false });
    data.append(FP8x23 { mag: 11835958, sign: false });
    data.append(FP8x23 { mag: 20491136, sign: false });
    data.append(FP8x23 { mag: 7553749, sign: false });
    data.append(FP8x23 { mag: 21703412, sign: false });
    data.append(FP8x23 { mag: 3402351, sign: false });
    data.append(FP8x23 { mag: 6197865, sign: false });
    data.append(FP8x23 { mag: 8319218, sign: false });
    data.append(FP8x23 { mag: 10348578, sign: false });
    data.append(FP8x23 { mag: 27789376, sign: true });
    data.append(FP8x23 { mag: 23445062, sign: true });
    data.append(FP8x23 { mag: 20440134, sign: true });
    data.append(FP8x23 { mag: 4380223, sign: true });
    data.append(FP8x23 { mag: 21481364, sign: true });
    data.append(FP8x23 { mag: 910452, sign: true });
    data.append(FP8x23 { mag: 9847078, sign: true });
    data.append(FP8x23 { mag: 11257807, sign: true });
    data.append(FP8x23 { mag: 19546126, sign: true });
    data.append(FP8x23 { mag: 8063282, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
