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
    data.append(FP16x16 { mag: 72575, sign: true });
    data.append(FP16x16 { mag: 22238, sign: false });
    data.append(FP16x16 { mag: 24075, sign: false });
    data.append(FP16x16 { mag: 41482, sign: true });
    data.append(FP16x16 { mag: 90068, sign: false });
    data.append(FP16x16 { mag: 16503, sign: false });
    data.append(FP16x16 { mag: 14909, sign: true });
    data.append(FP16x16 { mag: 47760, sign: false });
    data.append(FP16x16 { mag: 96266, sign: false });
    data.append(FP16x16 { mag: 5379, sign: true });
    data.append(FP16x16 { mag: 26424, sign: true });
    data.append(FP16x16 { mag: 62680, sign: true });
    data.append(FP16x16 { mag: 186270, sign: true });
    data.append(FP16x16 { mag: 107184, sign: false });
    data.append(FP16x16 { mag: 25050, sign: true });
    data.append(FP16x16 { mag: 105276, sign: false });
    data.append(FP16x16 { mag: 30285, sign: false });
    data.append(FP16x16 { mag: 86303, sign: false });
    data.append(FP16x16 { mag: 52763, sign: false });
    data.append(FP16x16 { mag: 29661, sign: true });
    data.append(FP16x16 { mag: 38831, sign: false });
    data.append(FP16x16 { mag: 125248, sign: true });
    data.append(FP16x16 { mag: 888, sign: true });
    data.append(FP16x16 { mag: 100186, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
