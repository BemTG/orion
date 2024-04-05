use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10033, sign: false });
    data.append(FP16x16 { mag: 47266, sign: false });
    data.append(FP16x16 { mag: 1589, sign: true });
    data.append(FP16x16 { mag: 21167, sign: true });
    data.append(FP16x16 { mag: 23912, sign: true });
    data.append(FP16x16 { mag: 19873, sign: true });
    data.append(FP16x16 { mag: 6234, sign: true });
    data.append(FP16x16 { mag: 5226, sign: true });
    data.append(FP16x16 { mag: 23267, sign: true });
    data.append(FP16x16 { mag: 1499, sign: true });
    data.append(FP16x16 { mag: 6847, sign: true });
    data.append(FP16x16 { mag: 106422, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
