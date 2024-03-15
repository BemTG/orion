use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42237, sign: true });
    data.append(FP16x16 { mag: 17243, sign: true });
    data.append(FP16x16 { mag: 68912, sign: true });
    data.append(FP16x16 { mag: 97905, sign: true });
    data.append(FP16x16 { mag: 56532, sign: false });
    data.append(FP16x16 { mag: 52015, sign: true });
    data.append(FP16x16 { mag: 36893, sign: false });
    data.append(FP16x16 { mag: 73441, sign: true });
    data.append(FP16x16 { mag: 65232, sign: false });
    data.append(FP16x16 { mag: 26353, sign: false });
    data.append(FP16x16 { mag: 48427, sign: false });
    data.append(FP16x16 { mag: 177446, sign: true });
    data.append(FP16x16 { mag: 127569, sign: false });
    data.append(FP16x16 { mag: 111302, sign: false });
    data.append(FP16x16 { mag: 55209, sign: false });
    data.append(FP16x16 { mag: 39474, sign: true });
    data.append(FP16x16 { mag: 130149, sign: false });
    data.append(FP16x16 { mag: 16143, sign: false });
    data.append(FP16x16 { mag: 60665, sign: false });
    data.append(FP16x16 { mag: 36573, sign: true });
    data.append(FP16x16 { mag: 116495, sign: false });
    data.append(FP16x16 { mag: 38734, sign: true });
    data.append(FP16x16 { mag: 25428, sign: false });
    data.append(FP16x16 { mag: 86105, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
