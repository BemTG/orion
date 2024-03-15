use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7837278, sign: false });
    data.append(FP8x23 { mag: 14348388, sign: true });
    data.append(FP8x23 { mag: 2624921, sign: true });
    data.append(FP8x23 { mag: 12126659, sign: true });
    data.append(FP8x23 { mag: 5621030, sign: true });
    data.append(FP8x23 { mag: 3149800, sign: false });
    data.append(FP8x23 { mag: 5157925, sign: false });
    data.append(FP8x23 { mag: 849361, sign: false });
    data.append(FP8x23 { mag: 2740211, sign: false });
    data.append(FP8x23 { mag: 9513125, sign: false });
    data.append(FP8x23 { mag: 6188311, sign: true });
    data.append(FP8x23 { mag: 7267785, sign: false });
    data.append(FP8x23 { mag: 7202642, sign: true });
    data.append(FP8x23 { mag: 1515372, sign: true });
    data.append(FP8x23 { mag: 4048458, sign: true });
    data.append(FP8x23 { mag: 13229346, sign: true });
    data.append(FP8x23 { mag: 10865753, sign: false });
    data.append(FP8x23 { mag: 5383268, sign: true });
    data.append(FP8x23 { mag: 8631375, sign: true });
    data.append(FP8x23 { mag: 2220104, sign: true });
    data.append(FP8x23 { mag: 3256011, sign: false });
    data.append(FP8x23 { mag: 7719686, sign: true });
    data.append(FP8x23 { mag: 11779878, sign: false });
    data.append(FP8x23 { mag: 13445092, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
