use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 47139, sign: true });
    data.append(FP16x16 { mag: 15096, sign: true });
    data.append(FP16x16 { mag: 6377, sign: false });
    data.append(FP16x16 { mag: 86535, sign: true });
    data.append(FP16x16 { mag: 69857, sign: false });
    data.append(FP16x16 { mag: 19452, sign: true });
    data.append(FP16x16 { mag: 46063, sign: true });
    data.append(FP16x16 { mag: 111305, sign: false });
    data.append(FP16x16 { mag: 916, sign: false });
    data.append(FP16x16 { mag: 49176, sign: true });
    data.append(FP16x16 { mag: 99724, sign: true });
    data.append(FP16x16 { mag: 26201, sign: false });
    data.append(FP16x16 { mag: 11835, sign: true });
    data.append(FP16x16 { mag: 26975, sign: true });
    data.append(FP16x16 { mag: 97585, sign: true });
    data.append(FP16x16 { mag: 5997, sign: true });
    data.append(FP16x16 { mag: 12783, sign: true });
    data.append(FP16x16 { mag: 14540, sign: false });
    data.append(FP16x16 { mag: 33130, sign: true });
    data.append(FP16x16 { mag: 147020, sign: false });
    data.append(FP16x16 { mag: 6953, sign: false });
    data.append(FP16x16 { mag: 9444, sign: false });
    data.append(FP16x16 { mag: 38131, sign: true });
    data.append(FP16x16 { mag: 100050, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
