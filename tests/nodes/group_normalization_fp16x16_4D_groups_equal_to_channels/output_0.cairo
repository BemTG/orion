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
    data.append(FP16x16 { mag: 27677, sign: false });
    data.append(FP16x16 { mag: 25412, sign: false });
    data.append(FP16x16 { mag: 8609, sign: false });
    data.append(FP16x16 { mag: 11485, sign: false });
    data.append(FP16x16 { mag: 55560, sign: false });
    data.append(FP16x16 { mag: 34953, sign: true });
    data.append(FP16x16 { mag: 20162, sign: false });
    data.append(FP16x16 { mag: 2417, sign: true });
    data.append(FP16x16 { mag: 20941, sign: false });
    data.append(FP16x16 { mag: 5081, sign: false });
    data.append(FP16x16 { mag: 28686, sign: false });
    data.append(FP16x16 { mag: 18475, sign: false });
    data.append(FP16x16 { mag: 3498, sign: false });
    data.append(FP16x16 { mag: 23130, sign: false });
    data.append(FP16x16 { mag: 51940, sign: false });
    data.append(FP16x16 { mag: 40217, sign: true });
    data.append(FP16x16 { mag: 8812, sign: false });
    data.append(FP16x16 { mag: 28772, sign: false });
    data.append(FP16x16 { mag: 12073, sign: false });
    data.append(FP16x16 { mag: 23525, sign: false });
    data.append(FP16x16 { mag: 27369, sign: false });
    data.append(FP16x16 { mag: 48822, sign: true });
    data.append(FP16x16 { mag: 26748, sign: false });
    data.append(FP16x16 { mag: 33055, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
