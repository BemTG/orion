use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Array<Tensor<FP8x23>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1596353, sign: false });
    data.append(FP8x23 { mag: 1385145, sign: false });
    data.append(FP8x23 { mag: 1385145, sign: false });
    data.append(FP8x23 { mag: 1385145, sign: false });
    data.append(FP8x23 { mag: 1385145, sign: false });
    data.append(FP8x23 { mag: 1385145, sign: false });
    data.append(FP8x23 { mag: 686313, sign: false });
    data.append(FP8x23 { mag: 686313, sign: false });
    data.append(FP8x23 { mag: 686313, sign: false });
    data.append(FP8x23 { mag: 686313, sign: false });
    data.append(FP8x23 { mag: 686313, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1816128, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 1480482, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });
    data.append(FP8x23 { mag: 731124, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
