use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(5);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 71168, sign: false });
    data.append(FP16x16 { mag: 72770, sign: false });
    data.append(FP16x16 { mag: 72551, sign: false });
    data.append(FP16x16 { mag: 69186, sign: false });
    data.append(FP16x16 { mag: 72950, sign: false });
    data.append(FP16x16 { mag: 48582, sign: true });
    data.append(FP16x16 { mag: 73035, sign: true });
    data.append(FP16x16 { mag: 107989, sign: true });
    data.append(FP16x16 { mag: 114820, sign: true });
    data.append(FP16x16 { mag: 38934, sign: true });
    data.append(FP16x16 { mag: 42925, sign: false });
    data.append(FP16x16 { mag: 46181, sign: false });
    data.append(FP16x16 { mag: 11416, sign: true });
    data.append(FP16x16 { mag: 16893, sign: false });
    data.append(FP16x16 { mag: 62054, sign: false });
    data.append(FP16x16 { mag: 43971, sign: false });
    data.append(FP16x16 { mag: 62008, sign: false });
    data.append(FP16x16 { mag: 42835, sign: false });
    data.append(FP16x16 { mag: 80394, sign: false });
    data.append(FP16x16 { mag: 67932, sign: false });
    data.append(FP16x16 { mag: 54221, sign: true });
    data.append(FP16x16 { mag: 2553, sign: false });
    data.append(FP16x16 { mag: 137786, sign: false });
    data.append(FP16x16 { mag: 47860, sign: false });
    data.append(FP16x16 { mag: 69243, sign: false });
    data.append(FP16x16 { mag: 69557, sign: false });
    data.append(FP16x16 { mag: 72310, sign: false });
    data.append(FP16x16 { mag: 71469, sign: false });
    data.append(FP16x16 { mag: 71602, sign: false });
    data.append(FP16x16 { mag: 73687, sign: false });
    data.append(FP16x16 { mag: 37536, sign: true });
    data.append(FP16x16 { mag: 118201, sign: true });
    data.append(FP16x16 { mag: 62214, sign: true });
    data.append(FP16x16 { mag: 107014, sign: true });
    data.append(FP16x16 { mag: 58394, sign: true });
    data.append(FP16x16 { mag: 59601, sign: false });
    data.append(FP16x16 { mag: 53358, sign: false });
    data.append(FP16x16 { mag: 18777, sign: false });
    data.append(FP16x16 { mag: 563, sign: true });
    data.append(FP16x16 { mag: 25463, sign: false });
    data.append(FP16x16 { mag: 76336, sign: false });
    data.append(FP16x16 { mag: 43992, sign: false });
    data.append(FP16x16 { mag: 75452, sign: false });
    data.append(FP16x16 { mag: 43061, sign: false });
    data.append(FP16x16 { mag: 58299, sign: false });
    data.append(FP16x16 { mag: 161413, sign: false });
    data.append(FP16x16 { mag: 1800, sign: true });
    data.append(FP16x16 { mag: 59621, sign: false });
    data.append(FP16x16 { mag: 35530, sign: true });
    data.append(FP16x16 { mag: 19519, sign: false });
    TensorTrait::new(shape.span(), data.span())
}