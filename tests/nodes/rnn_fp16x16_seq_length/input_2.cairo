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
    data.append(FP16x16 { mag: 2960, sign: false });
    data.append(FP16x16 { mag: 2046, sign: false });
    data.append(FP16x16 { mag: 7044, sign: false });
    data.append(FP16x16 { mag: 2435, sign: false });
    data.append(FP16x16 { mag: 9203, sign: false });
    data.append(FP16x16 { mag: 30, sign: false });
    data.append(FP16x16 { mag: 4157, sign: false });
    data.append(FP16x16 { mag: 3147, sign: false });
    data.append(FP16x16 { mag: 4571, sign: false });
    data.append(FP16x16 { mag: 5002, sign: false });
    data.append(FP16x16 { mag: 9874, sign: false });
    data.append(FP16x16 { mag: 3331, sign: false });
    data.append(FP16x16 { mag: 11186, sign: false });
    data.append(FP16x16 { mag: 10290, sign: false });
    data.append(FP16x16 { mag: 7993, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
