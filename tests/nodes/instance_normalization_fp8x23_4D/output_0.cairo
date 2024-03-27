use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 21991456, sign: false });
    data.append(FP8x23 { mag: 10897299, sign: false });
    data.append(FP8x23 { mag: 3875677, sign: false });
    data.append(FP8x23 { mag: 1679269, sign: false });
    data.append(FP8x23 { mag: 5745720, sign: false });
    data.append(FP8x23 { mag: 10343510, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
