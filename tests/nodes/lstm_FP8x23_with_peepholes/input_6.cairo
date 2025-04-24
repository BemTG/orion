use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_6() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(9);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 838860, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
