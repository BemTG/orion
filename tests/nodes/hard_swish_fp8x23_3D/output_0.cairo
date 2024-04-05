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
    data.append(FP8x23 { mag: 5283310, sign: false });
    data.append(FP8x23 { mag: 14341064, sign: false });
    data.append(FP8x23 { mag: 2468301, sign: true });
    data.append(FP8x23 { mag: 2911723, sign: true });
    data.append(FP8x23 { mag: 11276186, sign: false });
    data.append(FP8x23 { mag: 17849872, sign: false });
    data.append(FP8x23 { mag: 5087666, sign: false });
    data.append(FP8x23 { mag: 19949314, sign: false });
    data.append(FP8x23 { mag: 2542086, sign: true });
    data.append(FP8x23 { mag: 2588694, sign: true });
    data.append(FP8x23 { mag: 16121440, sign: false });
    data.append(FP8x23 { mag: 4762907, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
