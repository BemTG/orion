use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5862, sign: true });
    data.append(FP16x16 { mag: 27461, sign: false });
    data.append(FP16x16 { mag: 23948, sign: true });
    data.append(FP16x16 { mag: 99718, sign: false });
    data.append(FP16x16 { mag: 35908, sign: false });
    data.append(FP16x16 { mag: 33612, sign: false });
    data.append(FP16x16 { mag: 27753, sign: false });
    data.append(FP16x16 { mag: 26473, sign: false });
    data.append(FP16x16 { mag: 50581, sign: true });
    data.append(FP16x16 { mag: 77151, sign: false });
    data.append(FP16x16 { mag: 5593, sign: false });
    data.append(FP16x16 { mag: 65205, sign: false });
    data.append(FP16x16 { mag: 33725, sign: false });
    data.append(FP16x16 { mag: 26144, sign: false });
    data.append(FP16x16 { mag: 28008, sign: false });
    data.append(FP16x16 { mag: 35868, sign: false });
    data.append(FP16x16 { mag: 727, sign: true });
    data.append(FP16x16 { mag: 107360, sign: false });
    data.append(FP16x16 { mag: 38044, sign: false });
    data.append(FP16x16 { mag: 47308, sign: true });
    data.append(FP16x16 { mag: 34773, sign: false });
    data.append(FP16x16 { mag: 26834, sign: false });
    data.append(FP16x16 { mag: 31347, sign: false });
    data.append(FP16x16 { mag: 30791, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
