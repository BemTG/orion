use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 239770, sign: true });
    data.append(FP8x23 { mag: 1554163, sign: true });
    data.append(FP8x23 { mag: 280710, sign: false });
    data.append(FP8x23 { mag: 11240391, sign: false });
    data.append(FP8x23 { mag: 5105476, sign: false });
    data.append(FP8x23 { mag: 14059214, sign: false });
    data.append(FP8x23 { mag: 10692537, sign: true });
    data.append(FP8x23 { mag: 3975058, sign: true });
    data.append(FP8x23 { mag: 19799036, sign: true });
    data.append(FP8x23 { mag: 2091059, sign: true });
    data.append(FP8x23 { mag: 5291033, sign: true });
    data.append(FP8x23 { mag: 4909526, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
