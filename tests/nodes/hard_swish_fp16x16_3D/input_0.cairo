use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18353, sign: false });
    data.append(FP16x16 { mag: 69772, sign: false });
    data.append(FP16x16 { mag: 193375, sign: true });
    data.append(FP16x16 { mag: 61693, sign: true });
    data.append(FP16x16 { mag: 82152, sign: true });
    data.append(FP16x16 { mag: 141304, sign: true });
    data.append(FP16x16 { mag: 183229, sign: true });
    data.append(FP16x16 { mag: 185530, sign: true });
    data.append(FP16x16 { mag: 120986, sign: true });
    data.append(FP16x16 { mag: 193560, sign: true });
    data.append(FP16x16 { mag: 181798, sign: true });
    data.append(FP16x16 { mag: 128656, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
