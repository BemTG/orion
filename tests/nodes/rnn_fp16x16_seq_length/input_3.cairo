use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_3() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(10);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10202, sign: false });
    data.append(FP16x16 { mag: 5893, sign: false });
    data.append(FP16x16 { mag: 10969, sign: false });
    data.append(FP16x16 { mag: 2439, sign: false });
    data.append(FP16x16 { mag: 9859, sign: false });
    data.append(FP16x16 { mag: 7188, sign: false });
    data.append(FP16x16 { mag: 2941, sign: false });
    data.append(FP16x16 { mag: 13095, sign: false });
    data.append(FP16x16 { mag: 8687, sign: false });
    data.append(FP16x16 { mag: 4977, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
