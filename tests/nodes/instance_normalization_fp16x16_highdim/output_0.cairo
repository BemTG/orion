use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65327, sign: false });
    data.append(FP16x16 { mag: 58240, sign: false });
    data.append(FP16x16 { mag: 112509, sign: false });
    data.append(FP16x16 { mag: 83078, sign: false });
    data.append(FP16x16 { mag: 85472, sign: false });
    data.append(FP16x16 { mag: 125690, sign: false });
    data.append(FP16x16 { mag: 193196, sign: false });
    data.append(FP16x16 { mag: 166881, sign: false });
    data.append(FP16x16 { mag: 67542, sign: false });
    data.append(FP16x16 { mag: 11855, sign: false });
    data.append(FP16x16 { mag: 42287, sign: true });
    data.append(FP16x16 { mag: 29460, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
