use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16806912, sign: false });
    data.append(FP8x23 { mag: 13026855, sign: false });
    data.append(FP8x23 { mag: 20122830, sign: false });
    data.append(FP8x23 { mag: 24328122, sign: false });
    data.append(FP8x23 { mag: 10244346, sign: false });
    data.append(FP8x23 { mag: 11680887, sign: false });
    data.append(FP8x23 { mag: 2433778, sign: false });
    data.append(FP8x23 { mag: 7911818, sign: false });
    data.append(FP8x23 { mag: 11249044, sign: false });
    data.append(FP8x23 { mag: 21529216, sign: false });
    data.append(FP8x23 { mag: 19641030, sign: false });
    data.append(FP8x23 { mag: 21865430, sign: false });
    data.append(FP8x23 { mag: 4869210, sign: false });
    data.append(FP8x23 { mag: 5255287, sign: false });
    data.append(FP8x23 { mag: 14398988, sign: false });
    data.append(FP8x23 { mag: 7747343, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
