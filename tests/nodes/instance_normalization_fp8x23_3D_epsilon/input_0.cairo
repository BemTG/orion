use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2519933, sign: false });
    data.append(FP8x23 { mag: 12892399, sign: true });
    data.append(FP8x23 { mag: 7209204, sign: true });
    data.append(FP8x23 { mag: 1133624, sign: true });
    data.append(FP8x23 { mag: 16064325, sign: false });
    data.append(FP8x23 { mag: 11120935, sign: true });
    data.append(FP8x23 { mag: 7781104, sign: false });
    data.append(FP8x23 { mag: 10334058, sign: false });
    data.append(FP8x23 { mag: 11218000, sign: true });
    data.append(FP8x23 { mag: 12860922, sign: false });
    data.append(FP8x23 { mag: 2292188, sign: true });
    data.append(FP8x23 { mag: 6971177, sign: false });
    data.append(FP8x23 { mag: 5521836, sign: false });
    data.append(FP8x23 { mag: 4389916, sign: false });
    data.append(FP8x23 { mag: 3590608, sign: false });
    data.append(FP8x23 { mag: 7304364, sign: true });
    data.append(FP8x23 { mag: 10252519, sign: true });
    data.append(FP8x23 { mag: 6661299, sign: true });
    data.append(FP8x23 { mag: 1141264, sign: true });
    data.append(FP8x23 { mag: 7385935, sign: true });
    data.append(FP8x23 { mag: 9940967, sign: false });
    data.append(FP8x23 { mag: 9787109, sign: true });
    data.append(FP8x23 { mag: 3587379, sign: true });
    data.append(FP8x23 { mag: 10624075, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
