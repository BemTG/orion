use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 19142, sign: false });
    data.append(FP16x16 { mag: 110152, sign: true });
    data.append(FP16x16 { mag: 61797, sign: true });
    data.append(FP16x16 { mag: 151982, sign: true });
    data.append(FP16x16 { mag: 152433, sign: false });
    data.append(FP16x16 { mag: 171476, sign: true });
    data.append(FP16x16 { mag: 143641, sign: true });
    data.append(FP16x16 { mag: 5920, sign: false });
    data.append(FP16x16 { mag: 74591, sign: true });
    data.append(FP16x16 { mag: 105783, sign: true });
    data.append(FP16x16 { mag: 192616, sign: false });
    data.append(FP16x16 { mag: 8397, sign: true });
    data.append(FP16x16 { mag: 95325, sign: false });
    data.append(FP16x16 { mag: 145843, sign: false });
    data.append(FP16x16 { mag: 144759, sign: true });
    data.append(FP16x16 { mag: 113437, sign: true });
    data.append(FP16x16 { mag: 26889, sign: true });
    data.append(FP16x16 { mag: 178427, sign: true });
    data.append(FP16x16 { mag: 94330, sign: false });
    data.append(FP16x16 { mag: 32708, sign: false });
    data.append(FP16x16 { mag: 76982, sign: false });
    data.append(FP16x16 { mag: 187636, sign: true });
    data.append(FP16x16 { mag: 163980, sign: true });
    data.append(FP16x16 { mag: 36909, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
