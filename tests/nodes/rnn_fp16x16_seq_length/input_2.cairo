use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 69728, sign: true });
    data.append(FP16x16 { mag: 2289, sign: true });
    data.append(FP16x16 { mag: 93352, sign: true });
    data.append(FP16x16 { mag: 77392, sign: true });
    data.append(FP16x16 { mag: 39262, sign: true });
    data.append(FP16x16 { mag: 58088, sign: true });
    data.append(FP16x16 { mag: 28060, sign: false });
    data.append(FP16x16 { mag: 102472, sign: false });
    data.append(FP16x16 { mag: 75865, sign: true });
    data.append(FP16x16 { mag: 127781, sign: false });
    data.append(FP16x16 { mag: 20374, sign: false });
    data.append(FP16x16 { mag: 118537, sign: true });
    data.append(FP16x16 { mag: 35574, sign: false });
    data.append(FP16x16 { mag: 102045, sign: true });
    data.append(FP16x16 { mag: 21433, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
