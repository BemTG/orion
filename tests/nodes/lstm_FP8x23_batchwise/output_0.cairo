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
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 2799216, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 5221012, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });
    data.append(FP8x23 { mag: 6027877, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
