use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 77218, sign: false });
    data.append(FP16x16 { mag: 115321, sign: false });
    data.append(FP16x16 { mag: 60404, sign: false });
    data.append(FP16x16 { mag: 96971, sign: false });
    data.append(FP16x16 { mag: 152311, sign: false });
    data.append(FP16x16 { mag: 69703, sign: false });
    data.append(FP16x16 { mag: 116626, sign: false });
    data.append(FP16x16 { mag: 111118, sign: false });
    data.append(FP16x16 { mag: 28046, sign: false });
    data.append(FP16x16 { mag: 71699, sign: false });
    data.append(FP16x16 { mag: 33221, sign: false });
    data.append(FP16x16 { mag: 3563, sign: true });
    data.append(FP16x16 { mag: 173392, sign: true });
    data.append(FP16x16 { mag: 146996, sign: true });
    data.append(FP16x16 { mag: 77877, sign: true });
    data.append(FP16x16 { mag: 134424, sign: true });
    data.append(FP16x16 { mag: 11479, sign: false });
    data.append(FP16x16 { mag: 2481, sign: false });
    data.append(FP16x16 { mag: 53109, sign: true });
    data.append(FP16x16 { mag: 60010, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
