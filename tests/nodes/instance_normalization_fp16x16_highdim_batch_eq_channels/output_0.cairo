use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 119704, sign: true });
    data.append(FP16x16 { mag: 157802, sign: true });
    data.append(FP16x16 { mag: 28605, sign: true });
    data.append(FP16x16 { mag: 103396, sign: true });
    data.append(FP16x16 { mag: 73335, sign: true });
    data.append(FP16x16 { mag: 8264, sign: true });
    data.append(FP16x16 { mag: 58305, sign: true });
    data.append(FP16x16 { mag: 33834, sign: false });
    data.append(FP16x16 { mag: 69172, sign: true });
    data.append(FP16x16 { mag: 42482, sign: true });
    data.append(FP16x16 { mag: 95669, sign: false });
    data.append(FP16x16 { mag: 211626, sign: true });
    data.append(FP16x16 { mag: 52114, sign: true });
    data.append(FP16x16 { mag: 60821, sign: false });
    data.append(FP16x16 { mag: 33264, sign: true });
    data.append(FP16x16 { mag: 20496, sign: false });
    data.append(FP16x16 { mag: 111859, sign: true });
    data.append(FP16x16 { mag: 3390, sign: false });
    data.append(FP16x16 { mag: 143243, sign: true });
    data.append(FP16x16 { mag: 39279, sign: true });
    data.append(FP16x16 { mag: 67629, sign: true });
    data.append(FP16x16 { mag: 39269, sign: true });
    data.append(FP16x16 { mag: 158261, sign: true });
    data.append(FP16x16 { mag: 92546, sign: true });
    data.append(FP16x16 { mag: 21448, sign: false });
    data.append(FP16x16 { mag: 9634, sign: true });
    data.append(FP16x16 { mag: 56335, sign: true });
    data.append(FP16x16 { mag: 45416, sign: false });
    data.append(FP16x16 { mag: 13290, sign: false });
    data.append(FP16x16 { mag: 42810, sign: false });
    data.append(FP16x16 { mag: 33986, sign: true });
    data.append(FP16x16 { mag: 19561, sign: true });
    data.append(FP16x16 { mag: 170137, sign: true });
    data.append(FP16x16 { mag: 85473, sign: true });
    data.append(FP16x16 { mag: 95501, sign: false });
    data.append(FP16x16 { mag: 158829, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
