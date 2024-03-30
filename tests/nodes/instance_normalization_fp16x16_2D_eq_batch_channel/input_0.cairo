use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 145550, sign: true });
    data.append(FP16x16 { mag: 43268, sign: true });
    data.append(FP16x16 { mag: 55201, sign: false });
    data.append(FP16x16 { mag: 4054, sign: false });
    data.append(FP16x16 { mag: 145737, sign: true });
    data.append(FP16x16 { mag: 96298, sign: false });
    data.append(FP16x16 { mag: 45192, sign: false });
    data.append(FP16x16 { mag: 97352, sign: false });
    data.append(FP16x16 { mag: 4411, sign: false });
    data.append(FP16x16 { mag: 10605, sign: true });
    data.append(FP16x16 { mag: 6768, sign: true });
    data.append(FP16x16 { mag: 63923, sign: true });
    data.append(FP16x16 { mag: 1924, sign: true });
    data.append(FP16x16 { mag: 28271, sign: false });
    data.append(FP16x16 { mag: 116497, sign: true });
    data.append(FP16x16 { mag: 29178, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
