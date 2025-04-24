use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Array<Tensor<FP8x23>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(1);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 1469154, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });
    data.append(FP8x23 { mag: 816470, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
