use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 58714, sign: false });
    data.append(FP16x16 { mag: 100792, sign: false });
    data.append(FP16x16 { mag: 47531, sign: false });
    data.append(FP16x16 { mag: 44328, sign: true });
    data.append(FP16x16 { mag: 71010, sign: false });
    data.append(FP16x16 { mag: 119913, sign: false });
    data.append(FP16x16 { mag: 59904, sign: true });
    data.append(FP16x16 { mag: 48364, sign: true });
    data.append(FP16x16 { mag: 14983, sign: true });
    data.append(FP16x16 { mag: 82684, sign: false });
    data.append(FP16x16 { mag: 16447, sign: false });
    data.append(FP16x16 { mag: 7665, sign: true });
    data.append(FP16x16 { mag: 7190, sign: true });
    data.append(FP16x16 { mag: 43564, sign: true });
    data.append(FP16x16 { mag: 16474, sign: true });
    data.append(FP16x16 { mag: 19092, sign: true });
    data.append(FP16x16 { mag: 4779, sign: true });
    data.append(FP16x16 { mag: 5298, sign: false });
    data.append(FP16x16 { mag: 117951, sign: false });
    data.append(FP16x16 { mag: 13739, sign: true });
    data.append(FP16x16 { mag: 14847, sign: false });
    data.append(FP16x16 { mag: 130623, sign: true });
    data.append(FP16x16 { mag: 52082, sign: false });
    data.append(FP16x16 { mag: 32933, sign: true });
    data.append(FP16x16 { mag: 58600, sign: true });
    data.append(FP16x16 { mag: 50954, sign: true });
    data.append(FP16x16 { mag: 19982, sign: false });
    data.append(FP16x16 { mag: 43241, sign: false });
    data.append(FP16x16 { mag: 84680, sign: true });
    data.append(FP16x16 { mag: 94963, sign: true });
    data.append(FP16x16 { mag: 92494, sign: false });
    data.append(FP16x16 { mag: 59914, sign: false });
    data.append(FP16x16 { mag: 15675, sign: false });
    data.append(FP16x16 { mag: 47099, sign: false });
    data.append(FP16x16 { mag: 72890, sign: false });
    data.append(FP16x16 { mag: 25110, sign: false });
    data.append(FP16x16 { mag: 57383, sign: false });
    data.append(FP16x16 { mag: 120287, sign: true });
    data.append(FP16x16 { mag: 32126, sign: true });
    data.append(FP16x16 { mag: 163174, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
