use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 29821, sign: true });
    data.append(FP16x16 { mag: 99689, sign: true });
    data.append(FP16x16 { mag: 19275, sign: true });
    data.append(FP16x16 { mag: 52748, sign: true });
    data.append(FP16x16 { mag: 65402, sign: false });
    data.append(FP16x16 { mag: 19833, sign: false });
    data.append(FP16x16 { mag: 84864, sign: false });
    data.append(FP16x16 { mag: 168390, sign: false });
    data.append(FP16x16 { mag: 37430, sign: false });
    data.append(FP16x16 { mag: 87361, sign: true });
    data.append(FP16x16 { mag: 9470, sign: true });
    data.append(FP16x16 { mag: 77260, sign: true });
    data.append(FP16x16 { mag: 24038, sign: false });
    data.append(FP16x16 { mag: 12419, sign: true });
    data.append(FP16x16 { mag: 18981, sign: false });
    data.append(FP16x16 { mag: 50427, sign: true });
    data.append(FP16x16 { mag: 63164, sign: true });
    data.append(FP16x16 { mag: 27547, sign: false });
    data.append(FP16x16 { mag: 35921, sign: true });
    data.append(FP16x16 { mag: 74886, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
