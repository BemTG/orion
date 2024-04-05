use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2381272, sign: true });
    data.append(FP8x23 { mag: 11358074, sign: false });
    data.append(FP8x23 { mag: 19890674, sign: false });
    data.append(FP8x23 { mag: 14172564, sign: false });
    data.append(FP8x23 { mag: 12525247, sign: false });
    data.append(FP8x23 { mag: 242693, sign: true });
    data.append(FP8x23 { mag: 1782055, sign: true });
    data.append(FP8x23 { mag: 3354906, sign: false });
    data.append(FP8x23 { mag: 2267533, sign: true });
    data.append(FP8x23 { mag: 817827, sign: true });
    data.append(FP8x23 { mag: 1963373, sign: true });
    data.append(FP8x23 { mag: 3126122, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
