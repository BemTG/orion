use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 77721, sign: true });
    data.append(FP16x16 { mag: 113525, sign: true });
    data.append(FP16x16 { mag: 74182, sign: false });
    data.append(FP16x16 { mag: 58437, sign: true });
    data.append(FP16x16 { mag: 97843, sign: true });
    data.append(FP16x16 { mag: 54797, sign: true });
    data.append(FP16x16 { mag: 9592, sign: false });
    data.append(FP16x16 { mag: 220402, sign: true });
    data.append(FP16x16 { mag: 20802, sign: false });
    data.append(FP16x16 { mag: 65084, sign: false });
    data.append(FP16x16 { mag: 18698, sign: false });
    data.append(FP16x16 { mag: 185297, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
