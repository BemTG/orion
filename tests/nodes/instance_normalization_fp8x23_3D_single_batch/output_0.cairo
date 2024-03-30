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
    data.append(FP8x23 { mag: 8048669, sign: false });
    data.append(FP8x23 { mag: 10853620, sign: false });
    data.append(FP8x23 { mag: 26862928, sign: true });
    data.append(FP8x23 { mag: 22159492, sign: false });
    data.append(FP8x23 { mag: 954138, sign: true });
    data.append(FP8x23 { mag: 14024167, sign: false });
    data.append(FP8x23 { mag: 17833098, sign: false });
    data.append(FP8x23 { mag: 6832152, sign: false });
    data.append(FP8x23 { mag: 6409413, sign: false });
    data.append(FP8x23 { mag: 9343944, sign: false });
    data.append(FP8x23 { mag: 5362666, sign: false });
    data.append(FP8x23 { mag: 10720023, sign: false });
    data.append(FP8x23 { mag: 2191014, sign: true });
    data.append(FP8x23 { mag: 15700671, sign: false });
    data.append(FP8x23 { mag: 51754104, sign: false });
    data.append(FP8x23 { mag: 7474000, sign: false });
    data.append(FP8x23 { mag: 12384493, sign: true });
    data.append(FP8x23 { mag: 5675124, sign: true });
    data.append(FP8x23 { mag: 9382546, sign: true });
    data.append(FP8x23 { mag: 8421727, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
