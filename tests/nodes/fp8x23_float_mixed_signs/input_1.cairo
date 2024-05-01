use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 30810362, sign: false });
    data.append(FP8x23 { mag: 17198270, sign: true });
    data.append(FP8x23 { mag: 18350416, sign: true });
    data.append(FP8x23 { mag: 20574758, sign: false });
    data.append(FP8x23 { mag: 46282144, sign: true });
    data.append(FP8x23 { mag: 68770264, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
