use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 40008, sign: true });
    data.append(FP16x16 { mag: 21144, sign: true });
    data.append(FP16x16 { mag: 165620, sign: true });
    data.append(FP16x16 { mag: 87267, sign: true });
    data.append(FP16x16 { mag: 117276, sign: true });
    data.append(FP16x16 { mag: 16687, sign: true });
    data.append(FP16x16 { mag: 152152, sign: false });
    data.append(FP16x16 { mag: 86334, sign: false });
    data.append(FP16x16 { mag: 95907, sign: false });
    data.append(FP16x16 { mag: 2324, sign: false });
    data.append(FP16x16 { mag: 88665, sign: true });
    data.append(FP16x16 { mag: 31910, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
