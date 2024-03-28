use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 835429, sign: true });
    data.append(FP8x23 { mag: 4184407, sign: true });
    data.append(FP8x23 { mag: 688230, sign: false });
    data.append(FP8x23 { mag: 6514889, sign: false });
    data.append(FP8x23 { mag: 11429711, sign: true });
    data.append(FP8x23 { mag: 13803994, sign: false });
    data.append(FP8x23 { mag: 9774850, sign: false });
    data.append(FP8x23 { mag: 5482321, sign: true });
    data.append(FP8x23 { mag: 8100396, sign: true });
    data.append(FP8x23 { mag: 4925030, sign: false });
    data.append(FP8x23 { mag: 24642990, sign: false });
    data.append(FP8x23 { mag: 414118, sign: false });
    data.append(FP8x23 { mag: 8427536, sign: true });
    data.append(FP8x23 { mag: 6366033, sign: false });
    data.append(FP8x23 { mag: 7018001, sign: false });
    data.append(FP8x23 { mag: 20425300, sign: true });
    data.append(FP8x23 { mag: 6189996, sign: true });
    data.append(FP8x23 { mag: 6978809, sign: true });
    data.append(FP8x23 { mag: 3528110, sign: false });
    data.append(FP8x23 { mag: 5424584, sign: false });
    data.append(FP8x23 { mag: 7083834, sign: true });
    data.append(FP8x23 { mag: 8138287, sign: true });
    data.append(FP8x23 { mag: 5151880, sign: false });
    data.append(FP8x23 { mag: 4314710, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
