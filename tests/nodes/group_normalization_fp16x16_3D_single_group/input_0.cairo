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
    data.append(FP16x16 { mag: 14540, sign: false });
    data.append(FP16x16 { mag: 77028, sign: false });
    data.append(FP16x16 { mag: 116353, sign: true });
    data.append(FP16x16 { mag: 56299, sign: true });
    data.append(FP16x16 { mag: 86776, sign: false });
    data.append(FP16x16 { mag: 55196, sign: false });
    data.append(FP16x16 { mag: 7329, sign: false });
    data.append(FP16x16 { mag: 96594, sign: true });
    data.append(FP16x16 { mag: 10951, sign: true });
    data.append(FP16x16 { mag: 72231, sign: true });
    data.append(FP16x16 { mag: 87020, sign: true });
    data.append(FP16x16 { mag: 168377, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
