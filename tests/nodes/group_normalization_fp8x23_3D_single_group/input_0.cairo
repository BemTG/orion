use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 21390498, sign: false });
    data.append(FP8x23 { mag: 6181722, sign: true });
    data.append(FP8x23 { mag: 5453675, sign: false });
    data.append(FP8x23 { mag: 1063820, sign: true });
    data.append(FP8x23 { mag: 8632290, sign: false });
    data.append(FP8x23 { mag: 2537470, sign: false });
    data.append(FP8x23 { mag: 12599435, sign: false });
    data.append(FP8x23 { mag: 7617166, sign: true });
    data.append(FP8x23 { mag: 1294497, sign: false });
    data.append(FP8x23 { mag: 3803628, sign: true });
    data.append(FP8x23 { mag: 5682742, sign: false });
    data.append(FP8x23 { mag: 8305430, sign: true });
    data.append(FP8x23 { mag: 6001278, sign: true });
    data.append(FP8x23 { mag: 6107894, sign: true });
    data.append(FP8x23 { mag: 5158357, sign: false });
    data.append(FP8x23 { mag: 7493421, sign: true });
    data.append(FP8x23 { mag: 5715872, sign: false });
    data.append(FP8x23 { mag: 11617327, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
