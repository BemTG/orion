use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2077529, sign: false });
    data.append(FP8x23 { mag: 7161208, sign: false });
    data.append(FP8x23 { mag: 12558069, sign: false });
    data.append(FP8x23 { mag: 647235, sign: false });
    data.append(FP8x23 { mag: 12625249, sign: false });
    data.append(FP8x23 { mag: 13632507, sign: false });
    data.append(FP8x23 { mag: 33036244, sign: false });
    data.append(FP8x23 { mag: 1282309, sign: true });
    data.append(FP8x23 { mag: 12624399, sign: false });
    data.append(FP8x23 { mag: 13631985, sign: false });
    data.append(FP8x23 { mag: 12558096, sign: false });
    data.append(FP8x23 { mag: 647233, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
