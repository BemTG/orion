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
    data.append(FP8x23 { mag: 6379985, sign: true });
    data.append(FP8x23 { mag: 14435605, sign: false });
    data.append(FP8x23 { mag: 21467934, sign: false });
    data.append(FP8x23 { mag: 16940946, sign: false });
    data.append(FP8x23 { mag: 15501710, sign: false });
    data.append(FP8x23 { mag: 24670696, sign: true });
    data.append(FP8x23 { mag: 4298234, sign: true });
    data.append(FP8x23 { mag: 5505416, sign: false });
    data.append(FP8x23 { mag: 19231294, sign: true });
    data.append(FP8x23 { mag: 1758539, sign: true });
    data.append(FP8x23 { mag: 20297174, sign: true });
    data.append(FP8x23 { mag: 11589560, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
