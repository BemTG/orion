use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14240516, sign: false });
    data.append(FP8x23 { mag: 620566, sign: true });
    data.append(FP8x23 { mag: 5919673, sign: false });
    data.append(FP8x23 { mag: 3652775, sign: false });
    data.append(FP8x23 { mag: 2741481, sign: true });
    data.append(FP8x23 { mag: 4773652, sign: false });
    data.append(FP8x23 { mag: 13513484, sign: false });
    data.append(FP8x23 { mag: 7875029, sign: false });
    data.append(FP8x23 { mag: 2698872, sign: false });
    data.append(FP8x23 { mag: 7784562, sign: true });
    data.append(FP8x23 { mag: 7276119, sign: true });
    data.append(FP8x23 { mag: 3161972, sign: false });
    data.append(FP8x23 { mag: 1530599, sign: false });
    data.append(FP8x23 { mag: 9135041, sign: false });
    data.append(FP8x23 { mag: 3389896, sign: true });
    data.append(FP8x23 { mag: 2894224, sign: true });
    data.append(FP8x23 { mag: 987840, sign: false });
    data.append(FP8x23 { mag: 1053527, sign: true });
    data.append(FP8x23 { mag: 14788185, sign: false });
    data.append(FP8x23 { mag: 11077873, sign: true });
    data.append(FP8x23 { mag: 24603270, sign: false });
    data.append(FP8x23 { mag: 8719601, sign: false });
    data.append(FP8x23 { mag: 7181560, sign: true });
    data.append(FP8x23 { mag: 4733826, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
