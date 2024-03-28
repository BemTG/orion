use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54468, sign: true });
    data.append(FP16x16 { mag: 89182, sign: false });
    data.append(FP16x16 { mag: 43826, sign: false });
    data.append(FP16x16 { mag: 37900, sign: true });
    data.append(FP16x16 { mag: 84003, sign: true });
    data.append(FP16x16 { mag: 48198, sign: true });
    data.append(FP16x16 { mag: 86986, sign: true });
    data.append(FP16x16 { mag: 15318, sign: false });
    data.append(FP16x16 { mag: 13458, sign: true });
    data.append(FP16x16 { mag: 41078, sign: true });
    data.append(FP16x16 { mag: 78500, sign: false });
    data.append(FP16x16 { mag: 54789, sign: false });
    data.append(FP16x16 { mag: 103628, sign: true });
    data.append(FP16x16 { mag: 38808, sign: false });
    data.append(FP16x16 { mag: 48464, sign: true });
    data.append(FP16x16 { mag: 2252, sign: true });
    data.append(FP16x16 { mag: 13175, sign: true });
    data.append(FP16x16 { mag: 59815, sign: false });
    data.append(FP16x16 { mag: 61576, sign: true });
    data.append(FP16x16 { mag: 49538, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
