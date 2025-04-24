use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Array<Tensor<FP8x23>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 2443707, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 5069804, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });
    data.append(FP8x23 { mag: 6715072, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
