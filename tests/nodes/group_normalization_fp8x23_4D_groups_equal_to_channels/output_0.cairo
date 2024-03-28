use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7586656, sign: false });
    data.append(FP8x23 { mag: 4560466, sign: false });
    data.append(FP8x23 { mag: 1763387, sign: false });
    data.append(FP8x23 { mag: 7692204, sign: false });
    data.append(FP8x23 { mag: 2052780, sign: true });
    data.append(FP8x23 { mag: 1559569, sign: false });
    data.append(FP8x23 { mag: 619845, sign: true });
    data.append(FP8x23 { mag: 3992473, sign: true });
    data.append(FP8x23 { mag: 1894470, sign: false });
    data.append(FP8x23 { mag: 5104854, sign: false });
    data.append(FP8x23 { mag: 7057783, sign: false });
    data.append(FP8x23 { mag: 7545606, sign: false });
    data.append(FP8x23 { mag: 1109291, sign: true });
    data.append(FP8x23 { mag: 3615221, sign: true });
    data.append(FP8x23 { mag: 2519882, sign: true });
    data.append(FP8x23 { mag: 2138865, sign: false });
    data.append(FP8x23 { mag: 4628068, sign: false });
    data.append(FP8x23 { mag: 7960888, sign: false });
    data.append(FP8x23 { mag: 7070054, sign: false });
    data.append(FP8x23 { mag: 1943703, sign: false });
    data.append(FP8x23 { mag: 2717499, sign: true });
    data.append(FP8x23 { mag: 1920849, sign: false });
    data.append(FP8x23 { mag: 2502593, sign: true });
    data.append(FP8x23 { mag: 1806286, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
