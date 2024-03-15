use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9304191, sign: true });
    data.append(FP8x23 { mag: 7854164, sign: false });
    data.append(FP8x23 { mag: 9466705, sign: true });
    data.append(FP8x23 { mag: 2422423, sign: false });
    data.append(FP8x23 { mag: 124628, sign: true });
    data.append(FP8x23 { mag: 5811350, sign: true });
    data.append(FP8x23 { mag: 10531881, sign: true });
    data.append(FP8x23 { mag: 4442804, sign: true });
    data.append(FP8x23 { mag: 11993747, sign: true });
    data.append(FP8x23 { mag: 6757016, sign: true });
    data.append(FP8x23 { mag: 2867337, sign: false });
    data.append(FP8x23 { mag: 13341804, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
