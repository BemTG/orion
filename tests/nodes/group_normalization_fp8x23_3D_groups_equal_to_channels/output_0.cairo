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
    data.append(FP8x23 { mag: 2268731, sign: false });
    data.append(FP8x23 { mag: 1750992, sign: false });
    data.append(FP8x23 { mag: 5269735, sign: false });
    data.append(FP8x23 { mag: 10686392, sign: true });
    data.append(FP8x23 { mag: 2242386, sign: false });
    data.append(FP8x23 { mag: 1777337, sign: false });
    data.append(FP8x23 { mag: 11535990, sign: true });
    data.append(FP8x23 { mag: 6119333, sign: false });
    data.append(FP8x23 { mag: 2151207, sign: false });
    data.append(FP8x23 { mag: 1868516, sign: false });
    data.append(FP8x23 { mag: 9068844, sign: true });
    data.append(FP8x23 { mag: 3652185, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
