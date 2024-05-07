use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 76272960, sign: false });
    data.append(FP8x23 { mag: 87915952, sign: false });
    data.append(FP8x23 { mag: 44200540, sign: false });
    data.append(FP8x23 { mag: 75550392, sign: false });
    data.append(FP8x23 { mag: 31106968, sign: false });
    data.append(FP8x23 { mag: 61958532, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
