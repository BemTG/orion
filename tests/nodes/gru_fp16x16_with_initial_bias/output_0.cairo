use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 13142, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 10146, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });
    data.append(FP16x16 { mag: 4904, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}