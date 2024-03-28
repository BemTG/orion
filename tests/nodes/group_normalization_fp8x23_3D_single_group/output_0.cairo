use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1774189, sign: true });
    data.append(FP8x23 { mag: 2766609, sign: true });
    data.append(FP8x23 { mag: 5435634, sign: true });
    data.append(FP8x23 { mag: 2397718, sign: false });
    data.append(FP8x23 { mag: 37531992, sign: true });
    data.append(FP8x23 { mag: 8742143, sign: true });
    data.append(FP8x23 { mag: 12511089, sign: false });
    data.append(FP8x23 { mag: 19309014, sign: false });
    data.append(FP8x23 { mag: 63399, sign: true });
    data.append(FP8x23 { mag: 5705700, sign: false });
    data.append(FP8x23 { mag: 14058712, sign: true });
    data.append(FP8x23 { mag: 42317824, sign: false });
    data.append(FP8x23 { mag: 12330614, sign: false });
    data.append(FP8x23 { mag: 20210212, sign: false });
    data.append(FP8x23 { mag: 6645622, sign: false });
    data.append(FP8x23 { mag: 2048245, sign: false });
    data.append(FP8x23 { mag: 18888096, sign: true });
    data.append(FP8x23 { mag: 40416432, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
