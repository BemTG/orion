use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 29417, sign: false });
    data.append(FP16x16 { mag: 22963, sign: true });
    data.append(FP16x16 { mag: 47017, sign: false });
    data.append(FP16x16 { mag: 4758, sign: false });
    data.append(FP16x16 { mag: 100913, sign: true });
    data.append(FP16x16 { mag: 75367, sign: false });
    data.append(FP16x16 { mag: 103644, sign: true });
    data.append(FP16x16 { mag: 105339, sign: false });
    data.append(FP16x16 { mag: 13386, sign: true });
    data.append(FP16x16 { mag: 11016, sign: true });
    data.append(FP16x16 { mag: 48096, sign: true });
    data.append(FP16x16 { mag: 3627, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
